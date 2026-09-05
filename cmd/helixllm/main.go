package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/HelixDevelopment/HelixLLM/internal/agents"
	"github.com/HelixDevelopment/HelixLLM/internal/agents/tools"
	"github.com/HelixDevelopment/HelixLLM/internal/auth"
	"github.com/HelixDevelopment/HelixLLM/internal/brain"
	"github.com/HelixDevelopment/HelixLLM/internal/brain/models"
	"github.com/HelixDevelopment/HelixLLM/internal/control"
	"github.com/HelixDevelopment/HelixLLM/internal/fallback"
	"github.com/HelixDevelopment/HelixLLM/internal/gateway"
	gwmw "github.com/HelixDevelopment/HelixLLM/internal/gateway/middleware"
	"github.com/HelixDevelopment/HelixLLM/internal/knowledge"
	"github.com/HelixDevelopment/HelixLLM/internal/mode"
	"github.com/HelixDevelopment/HelixLLM/internal/server"
	"github.com/HelixDevelopment/HelixLLM/internal/shared/analytics"
	"github.com/HelixDevelopment/HelixLLM/internal/shared/config"
	"github.com/HelixDevelopment/HelixLLM/internal/shared/events"
	"github.com/HelixDevelopment/HelixLLM/internal/shared/hardware"
	"github.com/HelixDevelopment/HelixLLM/internal/shared/health"
	"github.com/HelixDevelopment/HelixLLM/internal/shared/i18n"
	"github.com/HelixDevelopment/HelixLLM/internal/shared/logging"
	"github.com/HelixDevelopment/HelixLLM/internal/shared/metrics"
	"github.com/HelixDevelopment/HelixLLM/internal/shared/observability"
)

func main() {
	modeFlag := flag.String("mode", "", "Operating mode (overrides HELIX_MODE env)")
	monitorFlag := flag.Bool("monitor", false, "Launch the TUI cluster status monitor instead of the server")
	challengesFlag := flag.Bool("challenges", false, "Run challenge banks instead of starting server")
	capabilityFlag := flag.Bool("capability", false, "Print the measured host capability profile as JSON and exit")
	banksDirFlag := flag.String("banks-dir", "challenges/banks/", "Directory containing YAML challenge banks")
	baseURLFlag := flag.String("base-url", "https://localhost:8443", "Base URL for challenge HTTP requests")
	categoryFlag := flag.String("category", "", "Run only challenges matching this category")
	priorityFlag := flag.String("priority", "", "Run only challenges matching this priority")
	caCertFlag := flag.String("ca-cert", "", "PEM file to trust for challenge HTTPS requests (e.g. certs/cert.pem for the self-signed dev server)")
	flag.Parse()

	// CONST-046: user-facing CLI strings resolved via i18n Translator.
	// Language selection follows the standard env-precedence fallback
	// (LANG / LC_ALL → "en") used by every CLI in the platform.
	lang := resolveCLILang()
	tr := i18n.New(lang)

	if *challengesFlag {
		os.Exit(runChallenges(tr, lang, *baseURLFlag, *banksDirFlag, *categoryFlag, *priorityFlag, *caCertFlag))
	}

	// --capability: dump the measured host capability profile and exit. This
	// runs before config.Load so it answers even on a host whose full
	// configuration is not set up — measurement is the question being asked.
	if *capabilityFlag {
		if err := printCapabilityProfile(context.Background(), os.Stdout); err != nil {
			fmt.Fprintf(os.Stderr, "%s\n", tr.T(lang, i18n.KeyHelixllmCLIGenericError, map[string]string{
				"detail": err.Error(),
			}))
			os.Exit(1)
		}
		return
	}

	cfg, err := config.Load()
	if err != nil {
		msg := tr.T(lang, i18n.KeyHelixllmCLIErrorLoadingConfig, map[string]string{
			"detail": err.Error(),
		})
		fmt.Fprintf(os.Stderr, "%s\n", msg)
		os.Exit(1)
	}

	// CLI flag overrides env
	if *modeFlag != "" {
		cfg.Mode = *modeFlag
	}

	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", tr.T(lang, i18n.KeyHelixllmCLIInvalidConfig, map[string]string{
			"detail": err.Error(),
		}))
		os.Exit(1)
	}

	m, err := mode.Parse(cfg.Mode)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", tr.T(lang, i18n.KeyHelixllmCLIGenericError, map[string]string{
			"detail": err.Error(),
		}))
		os.Exit(1)
	}

	log := logging.New(cfg.Log.Level, cfg.Log.Format)
	bus := events.NewBus()
	defer bus.Close()

	analyticsCollector := analytics.NewCollector(cfg.Analytics.ClickHouseAddr, cfg.Analytics.ClickHouseDatabase)
	defer analyticsCollector.Close() //nolint:errcheck

	obs, err := observability.New(observability.Options{
		ServiceName: "helixllm",
		Environment: "production",
		Exporter:    cfg.Log.OTELExporter,
	})
	if err != nil {
		log.Error(fmt.Sprintf("observability init failed: %v", err))
		os.Exit(1)
	}
	defer obs.Shutdown()

	// Register Prometheus metrics collectors.
	metrics.Register()

	// Build the Control Plane — needed by both the server and the monitor.
	cpOpts := control.ControlPlaneOptions{
		Hosts:    cfg.HostList(),
		SSHUser:  cfg.SSHUser,
		SSHKey:   cfg.SSHKey,
		Strategy: cfg.ScheduleStrategy,
	}
	if len(cfg.HostList()) > 0 {
		sshClient, sshErr := control.NewSSHClient(
			cfg.HostList()[0], 22, cfg.SSHUser, cfg.SSHKey,
		)
		if sshErr != nil {
			log.WithError(sshErr).Warn("SSH key unavailable; control plane running in no-op mode")
		} else {
			cpOpts.SSH = sshClient
		}
	}
	cp := control.NewControlPlane(cpOpts)

	// Graceful shutdown context — shared by both code paths.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Info("shutting down...")
		bus.Publish(events.TopicServerStopped, "main", nil)
		cancel()
	}()

	// --monitor: show the TUI cluster status display and exit.
	if *monitorFlag {
		log.Info("starting cluster monitor")
		if err := control.RunMonitor(ctx, cp, 2*time.Second); err != nil {
			log.WithError(err).Error("monitor error")
			os.Exit(1)
		}
		return
	}

	checker := health.NewChecker()

	log.WithField("mode", m.String()).Info("starting HelixLLM")

	srv := server.New(server.Options{
		Host:    cfg.Server.Host,
		Port:    cfg.Server.Port,
		TLSCert: cfg.Server.TLSCert,
		TLSKey:  cfg.Server.TLSKey,
		Checker: checker,
		Obs:     obs,
	})

	// Multi-model fleet initialization
	hwProfile := hardware.Detect()
	hardware.UpdateSystemMetrics(hwProfile)
	log.WithFields(map[string]interface{}{
		"gpu":         hwProfile.GPU.Available,
		"vram_mb":     hwProfile.GPU.VRAMTotal / (1024 * 1024),
		"preset":      hwProfile.PresetProfile,
		"inference":   cfg.LLM.InferenceMode,
		"numa_nodes":  len(hwProfile.NUMAIndices),
		"l3_cache_kb": hwProfile.L3CacheKB,
		"ram_gb":      hwProfile.RAM.Total / (1024 * 1024 * 1024),
	}).Info("Hardware detected")

	catalog := models.DefaultCatalog()
	var available []models.ModelDefinition
	if hwProfile.GPU.Available && cfg.LLM.InferenceMode != "cpu_only" {
		available = catalog.FilterByVRAM(hwProfile.GPU.VRAMFree)
	} else {
		available = catalog.FilterByCPUCompatible(hwProfile.RAM.Available)
		if cfg.LLM.InferenceMode == "cpu_only" && len(available) == 0 {
			log.Warn("No CPU-compatible models found; falling back to all chat models")
			available = catalog.ChatModels()
		}
	}

	log.WithField("count", len(available)).Info("Models selected for inference")

	registry := models.NewRegistry()
	downloader := brain.NewDownloader(cfg.LLM.ModelsDir)
	for _, def := range available {
		rm := models.RuntimeModel{
			Definition: def,
			Status:     models.StatusUnloaded,
			FilePath:   filepath.Join(cfg.LLM.ModelsDir, def.Filename),
			Downloaded: downloader.ModelExists(def.Filename),
		}
		registry.Add(rm)
	}

	// Pre-flight-resolve the embedded llama-server binary BEFORE spending any
	// time on model downloads. HXC-233: a prior deployment spent 37 minutes
	// downloading a model file for the embedded server to serve, then failed
	// to start the server because "llama-server" was not installed anywhere
	// on the host — the download was pure waste since nothing could ever
	// consume it. Resolving first means an absent binary is discovered
	// immediately, and the wasted download + doomed Start() attempt are both
	// skipped. HELIX_LLAMA_SERVER_EMBEDDED itself is left untouched (still
	// "enabled" from the operator's point of view) — only THIS run's
	// download+start work is skipped, so a later run started once the
	// binary is installed (or HELIX_LLAMA_SERVER_BINARY_PATH is pointed at
	// it) picks the embedded server back up with no further config change.
	llamaServerEmbedAvailable := false
	if cfg.LLM.LlamaServerEmbed {
		if _, resolveErr := brain.ResolveLlamaServerBinary(cfg.LLM.LlamaServerBinaryPath); resolveErr != nil {
			log.WithError(resolveErr).Warn("embedded llama-server binary not found — local model auto-download and embedded-server start are both skipped for this run; install llama-server (or set HELIX_LLAMA_SERVER_BINARY_PATH to its location) to restore local model serving, or set HELIX_LLAMA_SERVER_EMBEDDED=false to silence this warning if local serving is not needed")
		} else {
			llamaServerEmbedAvailable = true
		}
	}

	// Download missing models — only when something can actually consume
	// them this run (embedded server enabled AND its binary resolved above).
	if cfg.LLM.ModelsAutoDownload && llamaServerEmbedAvailable {
		for _, def := range available {
			if !downloader.ModelExists(def.Filename) {
				url := downloader.HuggingFaceURL(def.HuggingFaceRepo, def.Filename)
				log.WithField("model", def.ID).Info("Downloading model from HuggingFace")
				if err := downloader.Download(ctx, brain.DownloadRequest{URL: url, Filename: def.Filename}); err != nil {
					log.WithError(err).WithField("model", def.ID).Warn("Failed to download model, skipping")
				}
			}
		}
	}

	// Generate presets and start the embedded llama-server.
	var llamaSrv *brain.LlamaServer
	if llamaServerEmbedAvailable {
		var downloadedModels []models.ModelDefinition
		for _, def := range available {
			if downloader.ModelExists(def.Filename) {
				downloadedModels = append(downloadedModels, def)
			}
		}
		if len(downloadedModels) > 0 {
			presetsINI, _ := models.GeneratePresets(downloadedModels, hwProfile)
			presetsPath := filepath.Join(os.TempDir(), "helixllm-presets.ini")
			os.WriteFile(presetsPath, []byte(presetsINI), 0644) //nolint:errcheck

			threads := hwProfile.InferenceThreads()
			if cfg.LLM.CPUThreads > 0 {
				threads = cfg.LLM.CPUThreads
			}
			cpuOnly := !hwProfile.GPU.Available || cfg.LLM.InferenceMode == "cpu_only"
			llamaSrv = brain.NewLlamaServer(brain.LlamaServerConfig{
				BinaryPath:   cfg.LLM.LlamaServerBinaryPath,
				Port:         cfg.LLM.LlamaServerPort,
				ModelsDir:    cfg.LLM.ModelsDir,
				PresetsPath:  presetsPath,
				MaxModels:    cfg.LLM.ModelsMax,
				Threads:      threads,
				ThreadsBatch: hwProfile.BatchThreads(),
				NUMA:         hwProfile.NUMAEnabled && cfg.LLM.NUMAEnabled,
				MLock:        cfg.LLM.MLockEnabled,
				MMAP:         cfg.LLM.MMAPEnabled,
				NoKVOffload:  cpuOnly,
			})
			if err := llamaSrv.Start(ctx); err != nil {
				log.WithError(err).Warn("Failed to start embedded llama-server")
				// HXC-233: the process never launched — do NOT leave llamaSrv
				// pointing at a "server" whose port nothing is listening on.
				// The override below (LocalRPCHost/Port -> the embedded
				// server) must only fire when a real process is running,
				// otherwise every completion request is silently redirected
				// at a dead (or worse, unrelated-service-occupied) endpoint
				// instead of falling through to the router's real
				// no-provider-available error.
				llamaSrv = nil
			} else {
				log.Info("Waiting for llama-server to be ready...")
				if err := llamaSrv.WaitReady(ctx, 120*time.Second); err != nil {
					log.WithError(err).Warn("llama-server not ready within timeout")
				} else {
					for _, def := range downloadedModels {
						registry.UpdateStatus(def.ID, models.StatusLoaded)
					}
				}
			}
		}
	}
	if llamaSrv != nil {
		defer llamaSrv.Stop() //nolint:errcheck
		// Override to use embedded server
		cfg.LLM.LocalRPCHost = "127.0.0.1"
		cfg.LLM.LocalRPCPort = cfg.LLM.LlamaServerPort
	}

	// Create KV cache for conversation context persistence.
	//
	// HXC-244: redisKV stays in scope even when the gateway degrades to the
	// in-memory cache, so /internal/health can still probe the Redis the
	// operator CONFIGURED. Deriving the health component from whichever cache
	// object won would make a configured-but-unreachable Redis disappear from
	// the report entirely — silence in exactly the situation the operator most
	// needs to be told about.
	var kvCache brain.KVCacher
	var redisKV *brain.RedisKVCache
	var redisAddr string
	if cfg.Cache.RedisHost != "" {
		redisAddr = fmt.Sprintf("%s:%d", cfg.Cache.RedisHost, cfg.Cache.RedisPort)
		redisKV = brain.NewKVCache(brain.KVCacheConfig{
			RedisAddr:     redisAddr,
			RedisPassword: cfg.Cache.RedisPassword,
		})
		if redisKV.Available() {
			kvCache = redisKV
			log.Info("KV cache: Redis connected")
		} else {
			log.Warn("KV cache: Redis unreachable, falling back to in-memory")
			kvCache = brain.NewMemoryKVCache(time.Hour)
		}
	} else {
		kvCache = brain.NewMemoryKVCache(time.Hour)
		log.Info("KV cache: using in-memory (no Redis configured)")
	}

	// Create Brain — registers whichever providers are configured.
	brainSvc := brain.New(brain.Config{
		LlamaCppURL:       fmt.Sprintf("http://%s:%d", cfg.LLM.LocalRPCHost, cfg.LLM.LocalRPCPort),
		LlamaCppModels:    []string{cfg.LLM.LocalModel},
		OpenAIKey:         cfg.LLM.OpenAIKey,
		OpenAIBaseURL:     cfg.LLM.OpenAIBaseURL,
		AnthropicKey:      cfg.LLM.AnthropicKey,
		ChutesKey:         cfg.LLM.ChutesKey,
		OpenRouterKey:     cfg.LLM.OpenRouterKey,
		HuggingFaceKey:    cfg.LLM.HuggingFaceKey,
		NvidiaKey:         cfg.LLM.NvidiaKey,
		CerebrasKey:       cfg.LLM.CerebrasKey,
		SambaNovaKey:      cfg.LLM.SambaNovaKey,
		TogetherKey:       cfg.LLM.TogetherKey,
		DefaultProvider:   cfg.LLM.DefaultProvider,
		ComplexityEnabled: cfg.LLM.ComplexityEnabled,
		Registry:          registry,
		KVCache:           kvCache,
	})

	// Create embedder early so the gateway can use it for /v1/embeddings.
	embedder := buildEmbedder(cfg, log)

	// Create knowledge pipeline BEFORE gateway so RAG hook is available.
	// Host/port are env-configurable (§11.4.111 resolve-by-config) rather
	// than hardcoded, defaulting to localhost:6333 for zero-config local dev.
	store, err := knowledge.NewVectorStore(cfg.Knowledge.VectorDB, cfg.Knowledge.VectorDBHost, cfg.Knowledge.VectorDBPort)
	if err != nil {
		log.WithError(err).Error("failed to connect to vector store, falling back to memory store")
		store = knowledge.NewMemoryStore()
	}
	chunker := knowledge.NewFixedSizeChunker(cfg.Knowledge.RAGChunkSize, cfg.Knowledge.RAGChunkOverlap)

	// Cross-encoder reranking stage (embed -> retrieve -> RERANK -> ground),
	// config-gated per HELIX_RAG_RERANK_ENABLED. Wires a real TEI /rerank
	// endpoint (e.g. BAAI/bge-reranker-base served by
	// huggingface/text-embeddings-inference) into the production RAG query
	// path — previously this was proven only in a standalone QA harness and
	// never reached Pipeline.Query.
	var reranker knowledge.Reranker
	if cfg.Knowledge.RerankEnabled && cfg.Knowledge.RerankBaseURL != "" {
		reranker = knowledge.NewTEIReranker(cfg.Knowledge.RerankBaseURL)
		log.WithField("rerank_base_url", cfg.Knowledge.RerankBaseURL).Info("RAG cross-encoder reranking enabled")
	}

	pipeline := knowledge.NewPipeline(knowledge.PipelineConfig{
		Embedder:              embedder,
		Store:                 store,
		Chunker:               chunker,
		DefaultCollection:     "default",
		DefaultTopK:           cfg.Knowledge.RAGTopK,
		Reranker:              reranker,
		RerankFetchMultiplier: cfg.Knowledge.RerankFetchMultiplier,
	})

	// Build the FallbackChain — ordered by LLMsVerifier quality scores with
	// llamacpp always placed last as the local safety net.
	scorerBridge := fallback.NewScorerBridge(fallback.ScorerBridgeConfig{
		VerifierURL:     cfg.LLM.VerifierURL,
		RefreshInterval: parseDuration(cfg.LLM.ScoreRefreshInterval, 5*time.Minute),
	})

	// Discover models from all registered providers (must happen before
	// discoverProviderModels, which reads the cached model lists).
	fetchProviderModels(ctx, brainSvc)

	providerModels := discoverProviderModels(brainSvc)
	scores, _ := scorerBridge.FetchScores(ctx)
	entries := scorerBridge.BuildEntries(scores, providerModels)

	rateLimiter := fallback.NewRateLimitTracker(5, 1000)
	fallbackChain := newFallbackChain(brainSvc, entries, rateLimiter)

	scorerBridge.StartRefreshLoop(ctx, fallbackChain, providerModels)

	memAdapter := fallback.NewMemoryAdapter(fallback.MemoryAdapterConfig{
		HelixMemoryURL: cfg.LLM.MemoryURL,
		SyncInterval:   30 * time.Second,
		Enabled:        cfg.LLM.MemorySyncEnabled,
	})
	memAdapter.Start(ctx)

	for i, e := range entries {
		slog.Info("fallback chain entry",
			"rank", i+1,
			"provider", e.ProviderName,
			"model", e.ModelID,
			"score", e.Score,
			"local", e.IsLocalFallback,
		)
	}

	// HXC-244: publish the real dependency checks on /internal/health.
	//
	// Registration happens HERE rather than beside health.NewChecker() because
	// the dependencies being checked — the provider set, the KV cache, the
	// vector store — do not exist yet at that point. The server already holds
	// this same *Checker, and the aggregator's component list is mutex-guarded,
	// so registering now is safe; no request can observe the intermediate empty
	// list because srv.ListenAndServe is not called until the end of main.
	//
	// qdrantStore is nil when Qdrant was unreachable at startup (NewVectorStore
	// falls back to the in-process memory store), which is precisely the case
	// fallbackDependencyCheck reports as "configured but NOT in use".
	qdrantStore, _ := store.(*knowledge.QdrantStore)
	healthDeps := healthCheckDeps{
		Providers:     brainSvc.Providers,
		RedisAddr:     redisAddr,
		RedisInUse:    redisKV != nil && redisKV.Available(),
		VectorBackend: cfg.Knowledge.VectorDB,
		VectorTarget:  fmt.Sprintf("%s:%d", cfg.Knowledge.VectorDBHost, cfg.Knowledge.VectorDBPort),
		VectorInUse:   qdrantStore != nil,
		VerifierURL:   cfg.LLM.VerifierURL,
		HTTPClient:    &http.Client{Timeout: healthProbeTimeout},
	}
	if redisKV != nil {
		healthDeps.RedisProbe = redisKV.Ping
	}
	if qdrantStore != nil {
		healthDeps.VectorProbe = qdrantStore.Ping
	}
	registerHealthChecks(checker, healthDeps)

	// Register gateway routes with RAG hook — this injects retrieved codebase
	// context into every /v1/chat/completions request so small models have
	// relevant code pre-loaded instead of exploring via repeated tool calls.
	// The FallbackChain is the primary Completer; brainSvc is kept as
	// ModelBrain so /v1/models can enumerate available models.
	// Build the JWT credential (HELIX_AUTH_JWT_SECRET). An unset secret yields
	// a nil verifier — the documented off-switch — and every auth site then
	// behaves exactly as it did before JWT existed. cfg.Validate() has already
	// refused an unexpanded placeholder, a whitespace-only secret, and a
	// secret shorter than RFC 7518's HS256 minimum, so an error here would be
	// a bug in that ordering rather than an operator mistake; fail loudly
	// rather than boot with authentication in an unknown state.
	jwtVerifier, err := auth.New(cfg.Auth.JWTSecret, time.Duration(cfg.Auth.JWTTTLMinutes)*time.Minute)
	if err != nil {
		log.WithError(err).Error("refusing to start: HELIX_AUTH_JWT_SECRET is unusable")
		os.Exit(1)
	}
	logAuthPosture(log, cfg.Auth.APIKeys, jwtVerifier)

	toolMgr := gateway.DefaultToolManager()
	gateway.RegisterRoutes(srv.Router(), gateway.RouterOptions{
		APIKeys:         cfg.Auth.APIKeys,
		JWT:             jwtVerifier,
		RateLimit:       cfg.Server.RatePerMinute,
		Brain:           fallbackChain,
		ModelBrain:      brainSvc,
		Embedder:        embedder,
		ToolManager:     toolMgr,
		RAGHook:         knowledge.RAGHook(pipeline, "codebase"),
		TOONEnabled:     cfg.Features.TOON,
		HardwareProfile: hwProfile,
	})
	// DZ-05: gate the sensitive control/data/agent route groups with the SAME
	// middleware the gateway /v1 routes use. When neither credential is
	// configured the middleware runs in open-access mode — identical semantics
	// to the gateway /v1 group — so behaviour is unchanged for open
	// deployments and enforced the moment keys or a JWT secret are configured.
	//
	// Passing the SAME verifier instance the /v1 group got is the point: a
	// token minted by POST /v1/auth/token opens /internal/cluster/* and
	// /internal/knowledge/* too, and there is exactly one place where the
	// accepted-credential set is decided.
	dzAuth := gwmw.APIKeyOrJWTAuth(cfg.Auth.APIKeys, jwtVerifier)
	knowledge.RegisterKnowledgeRoutes(srv.Router(), pipeline, dzAuth)

	// Auto-ingest codebase if configured.
	if cfg.Knowledge.IngestDir != "" {
		ingestDir := cfg.Knowledge.IngestDir
		go func() {
			log.WithField("dir", ingestDir).Info("starting codebase auto-ingest")
			if err := knowledge.AutoIngest(pipeline, ingestDir, "codebase"); err != nil {
				log.WithError(err).Error("codebase auto-ingest failed")
			}
		}()
	}

	// Create tool registry with built-in tools.
	toolReg := agents.NewToolRegistry()
	toolReg.Register(&tools.EchoTool{})
	toolReg.Register(&tools.TimeTool{})
	toolReg.Register(tools.NewKnowledgeQueryTool(pipeline, "default"))

	// Security sandbox for file/exec/git/analysis tools.
	workDir, _ := os.Getwd()
	sandbox := tools.NewSandbox(tools.SandboxConfig{
		AllowedPaths: []string{"/tmp", workDir},
	})

	// File system tools (5).
	toolReg.Register(tools.NewReadFileTool(sandbox))
	toolReg.Register(tools.NewWriteFileTool(sandbox))
	toolReg.Register(tools.NewListDirectoryTool(sandbox))
	toolReg.Register(tools.NewSearchFilesTool(sandbox))
	toolReg.Register(tools.NewFileInfoTool(sandbox))

	// Code execution tools (2).
	toolReg.Register(tools.NewExecutePythonTool(sandbox))
	toolReg.Register(tools.NewExecuteShellTool(sandbox))

	// Git tools (8).
	toolReg.Register(tools.NewGitStatusTool(sandbox))
	toolReg.Register(tools.NewGitDiffTool(sandbox))
	toolReg.Register(tools.NewGitLogTool(sandbox))
	toolReg.Register(tools.NewGitBranchTool(sandbox))
	toolReg.Register(tools.NewGitCommitTool(sandbox))
	toolReg.Register(tools.NewGitPushTool(sandbox))
	toolReg.Register(tools.NewGitPullTool(sandbox))
	toolReg.Register(tools.NewGitCreateBranchTool(sandbox))

	// Analysis tools (4).
	toolReg.Register(tools.NewAnalyzeCodeTool(sandbox))
	toolReg.Register(tools.NewRunTestsTool(sandbox))
	toolReg.Register(tools.NewGetDependenciesTool(sandbox))
	toolReg.Register(tools.NewCalculateComplexityTool(sandbox))

	// Web tools (2).
	toolReg.Register(tools.NewWebSearchTool())
	toolReg.Register(tools.NewFetchURLTool())

	// Create agent with Brain, tools, and RAG hook.
	agentSvc := agents.NewAgent(agents.AgentConfig{
		Brain:    brainSvc,
		Tools:    toolReg,
		RAGHook:  knowledge.RAGHook(pipeline, "default"),
		MaxTurns: 10,
	})

	// Create conversation context for multi-turn sessions.
	convCtx := agents.NewConversationContext(100)

	// Create three-tier memory manager (working + episodic + semantic via RAG).
	memMgr := agents.NewMemoryManager(convCtx, pipeline)

	// Wire the MemoryAdapter into the MemoryManager so that high-importance
	// memories (importance >= 0.7) are queued for persistent sync to HelixMemory.
	if cfg.LLM.MemorySyncEnabled {
		memMgr.SetPersistentSyncer(memAdapter)
	}

	// Create multi-agent coordinator (phase-based: investigation → synthesis → implementation → verification).
	coordinator := agents.NewCoordinator(agents.CoordinatorConfig{
		Brain:   brainSvc,
		Tools:   toolReg,
		RAGHook: knowledge.RAGHook(pipeline, "default"),
	})

	// Create task planner (LLM-driven goal decomposition into ordered steps).
	planner := agents.NewPlanner(brainSvc)

	// Register agent routes (chat, tools, coordinate, plan, memory/remember, memory/recall, cache/stats).
	agents.RegisterAgentRoutesWithExtras(srv.Router(), agentSvc, convCtx, coordinator, planner, memMgr, kvCache, dzAuth)

	control.RegisterRoutes(srv.Router(), cp, dzAuth)

	bus.Publish(events.TopicServerStarted, "main", m.String())
	log.WithField("addr", fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)).
		Info("server listening")

	if err := srv.ListenAndServe(ctx); err != nil {
		log.WithError(err).Error("server error")
		os.Exit(1)
	}
}

// fetchProviderModels calls FetchModels on each provider that supports it,
// populating the cached model list before the fallback chain is built.
func fetchProviderModels(ctx context.Context, b *brain.Brain) {
	type modelFetcher interface {
		FetchModels(ctx context.Context, filterFn func(string) bool) error
	}
	type freeModelFetcher interface {
		DiscoverFreeModels(ctx context.Context) error
	}

	providers := b.Providers()
	slog.Info("fetchProviderModels: starting", "provider_count", len(providers))
	for name, p := range providers {
		slog.Info("fetchProviderModels: checking provider", "name", name, "type", fmt.Sprintf("%T", p))
		if f, ok := p.(freeModelFetcher); ok {
			if err := f.DiscoverFreeModels(ctx); err != nil {
				slog.Warn("failed to discover free models", "provider", name, "err", err)
			} else {
				slog.Info("discovered free models", "provider", name, "models", len(p.Models()))
			}
		} else if f, ok := p.(modelFetcher); ok {
			if err := f.FetchModels(ctx, nil); err != nil {
				slog.Warn("failed to fetch models", "provider", name, "err", err)
			} else {
				slog.Info("discovered models", "provider", name, "models", len(p.Models()))
			}
		}
	}
}

// buildEmbedder constructs the knowledge-layer Embedder from cfg.Knowledge,
// falling back to the deterministic HashEmbedder on construction error
// (unchanged behaviour from before this function was extracted).
//
// F07 (§11.4.146): HELIX_EMBEDDING_PROVIDER defaults to "local", which
// knowledge.NewEmbedder resolves to knowledge.HashEmbedder — a
// deterministic, NON-SEMANTIC embedder (SHA-256 bytes mapped into a
// unit-length vector) that does NOT capture any semantic similarity
// between texts. RAG retrieval built on it degrades to near-random
// ranking. That tradeoff was previously silent: an operator running with
// the (very common, zero-config) default would have no signal that their
// RAG pipeline is not doing semantic retrieval at all.
//
// This function does NOT change the default (that half is intentionally
// operator-gated per the task scope) — it only makes the choice
// transparent: whenever the RESOLVED embedder is the HashEmbedder (which
// covers "local", "hash", an empty value, an unrecognised provider name,
// AND the error-fallback path below — every path knowledge.NewEmbedder can
// take that ends in a HashEmbedder), a startup WARNING is logged. A
// type-assertion on the concrete returned value is used (rather than
// string-matching cfg.Knowledge.EmbeddingProvider) so the warning fires
// correctly for every one of those paths uniformly, including the ones
// that don't literally spell "local".
func buildEmbedder(cfg *config.HelixConfig, log logging.Logger) knowledge.Embedder {
	// For the "llama" embedding provider, the second argument is the
	// base URL of the embedding server. Use the dedicated
	// HELIX_EMBEDDING_BASE_URL when set; fall back to OpenAIBaseURL
	// for backward compatibility. For "openai" provider, it's the API key.
	embeddingAPIKeyOrURL := cfg.LLM.OpenAIKey
	if cfg.Knowledge.EmbeddingProvider == "llama" {
		if cfg.Knowledge.EmbeddingBaseURL != "" {
			embeddingAPIKeyOrURL = cfg.Knowledge.EmbeddingBaseURL
		} else if cfg.LLM.OpenAIBaseURL != "" {
			embeddingAPIKeyOrURL = cfg.LLM.OpenAIBaseURL
		}
	}
	embedder, err := knowledge.NewEmbedder(
		cfg.Knowledge.EmbeddingProvider,
		embeddingAPIKeyOrURL,
		cfg.Knowledge.EmbeddingModel,
		768,
	)
	if err != nil {
		log.WithError(err).Error("failed to create embedder, falling back to hash embedder")
		embedder = knowledge.NewHashEmbedder(768)
	}

	if _, isHash := embedder.(*knowledge.HashEmbedder); isHash {
		log.WithField("embedding_provider", cfg.Knowledge.EmbeddingProvider).
			Warn("RAG embeddings are using the non-semantic HashEmbedder " +
				"(HELIX_EMBEDDING_PROVIDER=local/hash, unset, or unrecognised, " +
				"or embedder construction failed) — embeddings do NOT capture " +
				"semantic similarity and RAG retrieval quality will be " +
				"significantly degraded; set HELIX_EMBEDDING_PROVIDER to a real " +
				"provider (e.g. \"openai\" or \"llama\" pointing at a real " +
				"embedding-serving endpoint) for production-quality RAG")
	}

	return embedder
}

// newFallbackChain builds the Chain that serves every /v1 request.
//
// It exists as a named function rather than three inline statements because the
// LAST of them is load-bearing and easy to lose: the chain is the Completer the
// gateway is wired to, and without the Brain as its model pinner it answers
// every request from its own top-ranked entry — so the identifier /v1/models
// publishes would resolve to nothing and a client asking for a specific model
// would be answered, confidently, by a different one. Tests build the serving
// stack through THIS function so a regression here cannot pass unnoticed.
func newFallbackChain(b *brain.Brain, entries []fallback.ChainEntry, rl *fallback.RateLimitTracker) *fallback.Chain {
	chain := fallback.NewChain(b.Providers(), rl)
	chain.SetEntries(entries)
	// Fill the naming registry BEFORE the chain starts serving. Resolution on
	// the request path is registry-only and performs no I/O (see
	// Brain.RegisterNames), so an identifier that was never registered resolves
	// to nothing — and a request naming it fails instead of reaching the model
	// it names. Registering here, rather than at some call site a future edit
	// might forget, ties it to the one place the pinner is installed: the
	// registry is filled for exactly the providers the chain will pin against.
	b.RegisterNames()
	// The Brain owns the naming registry and the provider set, so it is what
	// can say which provider a published identifier actually names.
	chain.SetModelPinner(b)
	return chain
}

// discoverProviderModels returns a map of provider name → first model ID by
// inspecting each registered provider.  The result is passed to ScorerBridge
// so chain entries carry a concrete model ID instead of an empty string.
func discoverProviderModels(b *brain.Brain) map[string]string {
	models := make(map[string]string)
	for name, p := range b.Providers() {
		ml := p.Models()
		if len(ml) > 0 {
			models[name] = ml[0]
		}
	}
	return models
}

// parseDuration parses s as a time.Duration.  If s is empty or unparseable it
// returns the provided fallback value.
func parseDuration(s string, fallbackVal time.Duration) time.Duration {
	if s == "" {
		return fallbackVal
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return fallbackVal
	}
	return d
}

// resolveCLILang picks a 2-letter language tag for user-facing CLI
// strings based on the standard POSIX env precedence (LC_ALL > LANG).
// Falls back to "en". Never returns "" — the i18n Translator's fallback
// chain assumes a non-empty default-language tag.
//
// CONST-046 round-95: hardcoded English literals at CLI call sites are
// replaced by i18n.Translator.T(lang, key); this function supplies lang.
func resolveCLILang() string {
	for _, env := range []string{"LC_ALL", "LANG"} {
		v := os.Getenv(env)
		if len(v) >= 2 {
			return v[:2]
		}
	}
	return "en"
}
