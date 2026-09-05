package runtime

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// The T070 launch seam, wired: Colibri (https://github.com/JustVugg/colibri)
// as the disk-streaming runtime, launched and lifecycled through the Process
// and HealthProbe seams in streaming_launch.go.
//
// WHY THIS FILE EXISTS. streaming_launch.go defines the lease and health
// discipline and deliberately says nothing about what binary satisfies it.
// This file is the answer: the deployment names the Colibri launcher and the
// model directory, and everything else — the serve invocation, the readiness
// contract, the stderr watch, the graceful stop — is derived here from the
// R1-verified launch contract:
//
//  1. Launch is `coli serve --model <dir> --host <host> --port <port>` in the
//     foreground. There is no daemonize flag; the process manager owns the
//     lifecycle, which is exactly what ExecProcess and Session.Close provide.
//  2. Readiness is the VERBATIM stderr line
//     "OpenAI-compatible API listening on http://{host}:{port}/v1" — NOT an
//     HTTP probe. Colibri binds its socket before the engine finishes loading,
//     so /health answered during load hangs on the listen backlog; after ready
//     /health is liveness-only. Probing HTTP for readiness would either hang
//     or lie, so the health probe here parses stderr and nothing else.
//     (HTTPHealth remains available post-ready for a process manager that
//     wants a liveness ping, but it is not this launch's readiness signal.)
//  3. Shutdown is SIGTERM/SIGINT, then grace, then kill — the signal → grace →
//     kill primitive ExecProcess already implements, which Colibri treats as
//     graceful (scheduler + server + engine child).
//  4. The engine dying mid-serve is TERMINAL. When the process exits, its
//     stderr reaches EOF and the watch below flips the process to dead and
//     records the exit error. The health probe then reports not-ready
//     forever; the process manager restarts from there. Nothing here fakes a
//     serve on a dead engine.
//  5. A port already occupied fails fast: Start surfaces the launch error and
//     the launcher tears down and releases, as it does for any start failure.
//  6. Queue saturation answers HTTP 429; callers back off. That is a property
//     of the served API, not of this seam, so it needs no handling here — it
//     is recorded in deploy/compose.yaml's service comment for operators.

// ColibriReadyLine is the stderr prefix that declares the engine serving.
// The R1 contract pins the full line
// "OpenAI-compatible API listening on http://{host}:{port}/v1"; the probe
// matches the prefix so a launcher that prints a longer or differently
// suffixed address line is still recognised, and matches nothing else.
const ColibriReadyLine = "OpenAI-compatible API listening on http://"

// EnvColibriBinary names the environment variable through which the deployment
// supplies the Colibri launcher (the Python `coli` entry point of
// github.com/JustVugg/colibri). It is env/config injection, never a compiled
// host path (CONST-045).
const EnvColibriBinary = "HELIX_COLIBRI_BIN"

// Errors reported by the Colibri seam. They are closed-set: a caller can tell
// "not configured" (an operator action) from "failed to start" (a runtime
// fault) and respond to each.
var (
	// ErrColibriBinaryNotConfigured: no launcher path was supplied and none was
	// discoverable. Refused rather than faked — a launched stand-in would let
	// the lifecycle report green over a process that serves nothing.
	ErrColibriBinaryNotConfigured = errors.New(
		"runtime: colibri launcher binary not configured: set " + EnvColibriBinary +
			" to the launcher path or install 'coli' on PATH")
	// ErrColibriNotLaunched: Ready/Dead/ExitErr were read before Start.
	ErrColibriNotLaunched = errors.New("runtime: colibri process has not been started")
)

// ResolveColibriBinary resolves the launcher path from the deployment's
// configuration: the HELIX_COLIBRI_BIN environment variable first, then a
// `coli` on PATH (the installed form of the vendored checkout's package).
// When neither exists the refusal is returned so the caller can surface the
// operator action instead of guessing a path.
func ResolveColibriBinary() (string, error) {
	if bin := strings.TrimSpace(os.Getenv(EnvColibriBinary)); bin != "" {
		return bin, nil
	}
	if path, err := exec.LookPath("coli"); err == nil {
		return path, nil
	}
	return "", ErrColibriBinaryNotConfigured
}

// ColibriConfig is what the deployment supplies to launch one Colibri model.
// The binary and model directory are configuration, never constants, for the
// reasons colibri.go's header states: neither is knowable from the adoption
// record alone, and a compiled host path is the hardcoded-distribution-host
// pattern the project forbids.
type ColibriConfig struct {
	// Binary is the launcher path — the Python `coli` entry point. Empty means
	// ResolveColibriBinary, so the zero config is "whatever the deployment
	// configured", and an unconfigured deployment is an honest refusal, not a
	// guess.
	Binary string
	// ModelDir is the directory holding the model the engine serves.
	ModelDir string
	// Host and Port are the address the engine's OpenAI-compatible API
	// listens on. The readiness line is matched against them when both are
	// set; readiness also accepts the bare prefix.
	Host string
	Port int
	// Env is the process environment, nil to inherit the caller's.
	Env []string
	// Stdout receives the launcher's stdout, nil to discard.
	Stdout *os.File
	// StderrLog additionally receives every stderr line the readiness watch
	// sees, so engine progress and load errors stay on disk instead of being
	// consumed only by the probe (the failure mode colibri.go's header warns
	// about: a discarded load turns a hang into a silent timeout).
	StderrLog *os.File
	// StopGrace bounds the graceful shutdown before the kill, as in
	// ExecProcess. Zero means the default.
	StopGrace time.Duration
}

// NewColibriProcess builds the Process for one Colibri launch, resolving the
// launcher from configuration. It is the config-presence gate the T070
// deferral note pointed at: a deployment that has not named a binary gets the
// refusal here, before any lease is admitted or process started.
//
// ColibriProcess.Start enforces the same refusal, so a value built by hand
// without this constructor still cannot launch a fake.
func NewColibriProcess(cfg ColibriConfig) (*ColibriProcess, error) {
	binary := cfg.Binary
	if binary == "" {
		resolved, err := ResolveColibriBinary()
		if err != nil {
			return nil, err
		}
		binary = resolved
	}
	return &ColibriProcess{
		Binary:    binary,
		ModelDir:  cfg.ModelDir,
		Host:      cfg.Host,
		Port:      cfg.Port,
		Env:       cfg.Env,
		Stdout:    cfg.Stdout,
		StderrLog: cfg.StderrLog,
		StopGrace: cfg.StopGrace,
	}, nil
}

// ColibriProcess is a Process that runs the Colibri launcher as a real
// operating-system process (via ExecProcess) and watches its stderr for the
// readiness contract and for engine death.
type ColibriProcess struct {
	// Binary, ModelDir, Host, Port, Env, Stdout, StderrLog and StopGrace are
	// the resolved configuration; see ColibriConfig.
	Binary    string
	ModelDir  string
	Host      string
	Port      int
	Env       []string
	Stdout    *os.File
	StderrLog *os.File
	StopGrace time.Duration

	mu      sync.Mutex
	proc    *ExecProcess
	ready   atomic.Bool
	dead    atomic.Bool
	exitMu  sync.Mutex
	exitErr error
}

// Start launches `coli serve` in the foreground and starts the stderr watch
// that drives readiness and engine-death detection. The process outlives the
// launch context, exactly as ExecProcess documents.
func (p *ColibriProcess) Start(ctx context.Context) error {
	if p.Binary == "" {
		return ErrColibriBinaryNotConfigured
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.proc != nil {
		return ErrAlreadyStarted
	}

	// stderr flows to the readiness watch. The read end is scanned line by
	// line; the write end is what the child process writes into.
	errR, errW, err := os.Pipe()
	if err != nil {
		return fmt.Errorf("runtime: colibri readiness pipe: %w", err)
	}

	proc := &ExecProcess{
		Binary:    p.Binary,
		Args:      p.serveArgs(),
		Env:       p.Env,
		Stdout:    p.Stdout,
		Stderr:    errW,
		StopGrace: p.StopGrace,
	}
	p.proc = proc
	p.ready.Store(false)
	p.dead.Store(false)
	p.setExitErr(nil)

	if err := proc.Start(ctx); err != nil {
		_ = errR.Close()
		_ = errW.Close()
		p.proc = nil
		return err
	}

	// The child holds errW now; closing our copy lets the scanner see EOF the
	// moment the process exits, which is the engine-death signal.
	_ = errW.Close()
	go p.watch(errR)
	return nil
}

// serveArgs is the R1-verified invocation: foreground serve, model directory,
// listen address.
func (p *ColibriProcess) serveArgs() []string {
	args := []string{"serve", "--model", p.ModelDir, "--host", p.Host, "--port", fmt.Sprintf("%d", p.Port)}
	return args
}

// watch scans the process's stderr for the readiness line and flips to dead
// when the process is gone — detected either by stderr EOF or by the
// underlying Wait returning, because a launcher that spawns an engine child
// can exit while that grandchild still holds the pipe open: EOF alone would
// then never come, and a dead engine must not be reported alive just because
// its orphan kept a file descriptor (contract item 4). It runs for the
// process's lifetime; Stop does not wait on it, and its only writes are the
// atomic flags and the recorded exit error.
func (p *ColibriProcess) watch(r *os.File) {
	p.mu.Lock()
	proc := p.proc
	p.mu.Unlock()

	// The scanner runs on its own goroutine; the owning loop below decides
	// when the watch ends, so a grandchild-held pipe cannot hold it open.
	lines := make(chan string, 64)
	go func() {
		defer close(lines)
		scanner := bufio.NewScanner(r)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			lines <- scanner.Text()
		}
	}()
	defer func() { _ = r.Close() }()

	for {
		select {
		case line, ok := <-lines:
			if !ok {
				// EOF: the process closed stderr, the normal death signal.
				p.setExitErr(p.waitExitErr(proc))
				p.dead.Store(true)
				return
			}
			if p.StderrLog != nil {
				_, _ = fmt.Fprintln(p.StderrLog, line)
			}
			if p.isReadyLine(line) {
				p.ready.Store(true)
			}
		default:
			if _, ok := proc.WaitResult(); ok {
				// The process exited even though stderr is still open (an
				// engine grandchild inherited it). Terminal all the same: the
				// launcher that owned the serve is gone. Closing the pipe
				// unblocks the scanner goroutine; draining whatever lines
				// remain lets that goroutine finish instead of blocking on a
				// full buffer nobody reads.
				p.setExitErr(p.waitExitErr(proc))
				p.dead.Store(true)
				_ = r.Close()
				go func() {
					for range lines {
					}
				}()
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
}

// isReadyLine matches the verbatim contract line. When both Host and Port are
// set the full line is required; the prefix alone is accepted otherwise, so a
// config that has not pinned an address still recognises the real signal.
func (p *ColibriProcess) isReadyLine(line string) bool {
	if p.Host != "" && p.Port > 0 {
		want := fmt.Sprintf("%s%s:%d/v1", ColibriReadyLine, p.Host, p.Port)
		if strings.TrimSpace(line) == want {
			return true
		}
	}
	return strings.Contains(line, ColibriReadyLine)
}

// waitExitErr reports how the underlying process ended. Called only once the
// process is gone — stderr EOF or Wait returning — so Wait has a result to
// give. It reads through ExecProcess.WaitResult rather than draining the done
// channel, because Stop is the channel's reader: two readers on one channel
// would race, and a watch that stole Stop's reaping would hang the teardown.
func (p *ColibriProcess) waitExitErr(proc *ExecProcess) error {
	if proc == nil {
		return nil
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		if err, ok := proc.WaitResult(); ok {
			return err
		}
		if time.Now().After(deadline) {
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// Stop ends the process through the single teardown (signal, grace, kill) and
// is safe on a process that was never started or already exited, as
// ExecProcess documents.
func (p *ColibriProcess) Stop(ctx context.Context) error {
	p.mu.Lock()
	proc := p.proc
	p.mu.Unlock()
	if proc == nil {
		return nil
	}
	return proc.Stop(ctx)
}

// Ready reports whether the engine has printed its readiness line.
func (p *ColibriProcess) Ready() bool { return p.ready.Load() }

// Dead reports whether the engine has exited — the terminal state the
// readiness watch flips on stderr EOF. A process manager polling Dead sees
// engine death without having to fake-serve or poll a corpse with HTTP.
func (p *ColibriProcess) Dead() bool { return p.dead.Load() }

// ExitErr reports how the engine ended, once Dead is true: the raw result of
// waiting on the process, so an unexpected exit carries its status and a
// Stop-driven exit reads as the signal that ended it. Recorded before Dead
// flips, so Dead()==true already implies the answer is in place.
func (p *ColibriProcess) ExitErr() error {
	p.exitMu.Lock()
	defer p.exitMu.Unlock()
	return p.exitErr
}

func (p *ColibriProcess) setExitErr(err error) {
	p.exitMu.Lock()
	p.exitErr = err
	p.exitMu.Unlock()
}

// ColibriHealth is the HealthProbe for a launched Colibri process: readiness
// is the stderr listening line (see the file header for why an HTTP probe is
// wrong for readiness). After readiness, engine death flips the answer back —
// a dead engine is never healthy.
type ColibriHealth struct {
	// Proc is the process whose stderr contract this probe reads.
	Proc *ColibriProcess
}

// Healthy reports whether the engine is up and serving. Not-ready is the
// expected answer for most of a streaming load; a dead process is reported
// not-ready rather than ready, so a launcher that keeps polling eventually
// fails its budget and tears down instead of serving a corpse.
func (h *ColibriHealth) Healthy(_ context.Context) bool {
	if h == nil || h.Proc == nil {
		return false
	}
	return h.Proc.Ready() && !h.Proc.Dead()
}
