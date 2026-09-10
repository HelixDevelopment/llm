package fallback

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/HelixDevelopment/HelixLLM/internal/brain"
	"github.com/HelixDevelopment/HelixLLM/pkg/types"
)

// ErrProvidersExhausted marks the condition "no configured provider could
// serve this request" — every entry was skipped, unavailable, or failed.
//
// It exists so a caller can DISTINGUISH that availability condition from an
// ordinary upstream fault without matching on message text. The distinction
// is load-bearing at the HTTP boundary: an exhausted chain is a 503 (the
// service cannot serve right now, retry with backoff), while a provider that
// answered with a genuine fault is a 500. Callers that cannot tell them apart
// report both as 500, which tells clients, load balancers, and readiness
// probes that a warming-up backend is a broken build.
//
// Its message text is the historical prefix verbatim, because chain_test.go
// and Chain.Complete's own doc comment both assert on that substring.
var ErrProvidersExhausted = errors.New("all providers exhausted")

// IsProvidersExhausted reports whether err (or anything it wraps) is the
// exhausted-chain condition. Prefer this over errors.Is at call sites so the
// sentinel stays an implementation detail of this package.
func IsProvidersExhausted(err error) bool {
	return errors.Is(err, ErrProvidersExhausted)
}

// ErrPinnedModelUnavailable marks the condition "the SPECIFIC model this
// request named cannot be served right now" — the pinned path's sibling of
// ErrProvidersExhausted.
//
// It exists for the same reason: the two conditions are the same KIND of
// answer, so they must reach the client as the same status. A request whose
// named model has no serving provider, or whose serving provider is down, is
// an availability condition — 503, retry with backoff — not an internal fault.
// Reporting it as 500 tells every client, load balancer and readiness probe
// that the build is broken and that retrying is pointless.
//
// It is deliberately NOT ErrProvidersExhausted: nothing was exhausted, because
// the pinned path never iterates the chain. Keeping them distinct lets a caller
// tell "your model is not being served" from "none of ours are", while
// [IsUnservable] answers the question the HTTP boundary actually asks.
var ErrPinnedModelUnavailable = errors.New("pinned model unavailable")

// ErrRetiredIdentifier marks the condition "this request named an identifier
// this deployment has permanently stopped publishing".
//
// It is deliberately NOT wrapped in ErrPinnedModelUnavailable, because it is
// the opposite kind of answer. ErrPinnedModelUnavailable says "not right now,
// retry with backoff"; this says "never again, and here is what to do instead".
// A client obeying a retry-with-backoff instruction against a name that will
// never resolve retries forever, which is the same defect this project accepted
// for malformed requests: a retry that cannot succeed is a correct client
// looping indefinitely because the server told it to.
//
// The distinction is only available for a BOUNDED set. A published identifier
// carries a digest, so the host generally cannot be recovered from it and a
// re-minted name is indistinguishable from a host that is down — which is why
// every other unresolvable identifier stays ErrPinnedModelUnavailable. The
// retired loopback renderings are the exception: their host segment is readable
// and this project knows it stopped emitting them. See
// naming.Ruleset.HasRetiredHostSegment.
var ErrRetiredIdentifier = errors.New("identifier retired by a serving-host rename")

// IsRetiredIdentifier reports whether err (or anything it wraps) is the
// retired-identifier condition. The HTTP boundary asks this BEFORE
// [IsUnservable], because the two answers are mutually exclusive and only one
// of them tells the client to retry.
func IsRetiredIdentifier(err error) bool {
	return errors.Is(err, ErrRetiredIdentifier)
}

// IsUnservable reports whether err is an availability condition — the request
// cannot be served RIGHT NOW — rather than a fault in a provider that answered.
// It is the single question the HTTP boundary asks, so both sentinels are
// checked in one place instead of at each call site.
//
// A retired identifier is NOT unservable: nothing about it is temporary.
func IsUnservable(err error) bool {
	return IsProvidersExhausted(err) || errors.Is(err, ErrPinnedModelUnavailable)
}

// Chain is the central fallback orchestrator.  It holds an ordered list of
// ChainEntry values and routes each request to the first available provider,
// automatically failing over on 429 / 5xx errors.
//
// Entries are tried in the order stored in the slice.  Local-fallback entries
// (IsLocalFallback=true) should be placed at the end of the slice by the
// caller (e.g. ScorerBridge) so cloud providers are always attempted first.
type Chain struct {
	providers   map[string]brain.Provider
	entries     []ChainEntry
	rateLimiter *RateLimitTracker
	pinner      ModelPinner
	mu          sync.RWMutex
}

// ModelPinner resolves a requested model name to the provider that actually
// serves it.
//
// It exists because the Chain cannot answer that question on its own: it holds
// providers and scores, while the mapping from a published identifier to a
// served model lives in the Brain's naming registry. The Chain sees only this
// interface so it never has to know how identifiers are derived. *brain.Brain
// satisfies it.
type ModelPinner interface {
	// PinModel reports whether the request named a specific served model and,
	// if so, which provider answers to it under which name. See
	// brain.Brain.PinModel for the meaning of the three outcomes — in
	// particular, ok=true with an empty provider means the caller named one of
	// our identifiers whose host is not serving, which is an error rather than
	// an invitation to substitute.
	PinModel(requested string) (provider, model string, ok bool)

	// IsRetiredIdentifier reports whether requested is one of the identifiers
	// this deployment has permanently stopped publishing, as opposed to one it
	// simply cannot serve right now.
	//
	// It arrives through this interface, alongside the pin, because the answer
	// depends on which hosts the naming registry currently holds identities
	// for — a live machine can be called `localhost.lan`, whose identifiers
	// open with the same segment a retired one does. The Chain has no registry,
	// so deciding it here from the name alone told holders of live identifiers
	// that their model was gone for good. See brain.Brain.IsRetiredIdentifier.
	IsRetiredIdentifier(requested string) bool
}

// NewChain returns a Chain backed by the given provider map and rate limiter.
// providers maps brain.Provider.Name() → Provider instance.
// rl must not be nil.
func NewChain(providers map[string]brain.Provider, rl *RateLimitTracker) *Chain {
	return &Chain{
		providers:   providers,
		rateLimiter: rl,
	}
}

// SetEntries replaces the ordered entry list.  It is safe to call concurrently.
func (c *Chain) SetEntries(entries []ChainEntry) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = entries
}

// SetModelPinner installs the resolver consulted before the entry list, so a
// request naming a served model reaches THAT model instead of the chain's own
// per-entry default. It is safe to call concurrently.
//
// Leaving it nil keeps the pure score-ordered behaviour, which is what a caller
// that has no naming registry (most tests) wants.
func (c *Chain) SetModelPinner(p ModelPinner) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.pinner = p
}

// pin asks the installed pinner about a requested model name.
func (c *Chain) pin(requested string) (provider, model string, ok bool) {
	c.mu.RLock()
	p := c.pinner
	c.mu.RUnlock()
	if p == nil {
		return "", "", false
	}
	return p.PinModel(requested)
}

// refuseUnknownModel reports the error owed to a request whose model name the
// pin could not resolve, or nil when the score-ordered chain may answer it.
//
// # What the pin's miss means
//
// Brain.PinModel documents three outcomes, and ok=false is "nothing
// registered serves this name". Two very different requests arrive there: one
// that named NO model, and one that named a model this deployment does not
// serve. The chain used to treat both as "the caller named nothing" and hand
// them to its own top-ranked entry — which for the first is the feature, and
// for the second is a silent misroute. Measured on the shipped gateway,
// `definitely-not-a-real-model-zzz`, `gpt-4o` and `claude-opus-4` each
// returned HTTP 200 answered by the single local model, with nothing in the
// response a client could use to detect the substitution.
//
// This is the same trade pinnedProvider already makes one branch away, and
// for the same stated reason: "a caller who names one and cannot have it is
// better served by an error it can see."
//
// # Why a missing pinner is not a refusal
//
// Without a pinner the chain has no registry and therefore no basis to call a
// name unknown — exactly the reasoning [Chain.retired] already applies. A
// Chain wired without one keeps its pre-existing score-ordered behaviour,
// which is what every caller that has no naming registry (most tests, and any
// embedder using the Chain standalone) depends on.
// # Why a delegation sentinel is not a refusal
//
// The empty string was originally the only exemption here, on the grounds
// that it means "the caller named nothing". `auto` means exactly that too —
// it is the project's documented "you choose" value and the one its own
// canonical first API call sends — so refusing it broke a documented client
// flow while fixing an undocumented one. brain.IsDelegationSentinel is the
// single definition of that set, shared with Router.Route so the two refusal
// sites cannot drift into disagreeing about which requests name a model.
func (c *Chain) refuseUnknownModel(requested string) error {
	if brain.IsDelegationSentinel(requested) {
		return nil
	}
	c.mu.RLock()
	p := c.pinner
	c.mu.RUnlock()
	if p == nil {
		return nil
	}
	return brain.NewModelNotFound(requested)
}

// retired asks the installed pinner whether a requested name is one this
// deployment has permanently stopped publishing.
//
// With no pinner installed the answer is false: a Chain that cannot resolve
// names has no basis to call one permanently gone, and the retryable answer is
// the safe one.
func (c *Chain) retired(requested string) bool {
	c.mu.RLock()
	p := c.pinner
	c.mu.RUnlock()
	if p == nil {
		return false
	}
	return p.IsRetiredIdentifier(requested)
}

// pinnedProvider returns the provider a pinned request must be served by, or an
// error naming what the caller asked for.
//
// It deliberately does NOT fall through to the rest of the chain. A published
// identifier names one model on one host; a raw model name a provider serves is
// the same explicit choice written the older way. Answering either from some
// other provider is precisely the silent misroute the pin exists to prevent —
// the caller gets a confident response from a model it never asked for and has
// no way to detect the substitution. A caller who genuinely wants
// any-available-model names no model at all and gets the score-ordered chain;
// a caller who names one and cannot have it is better served by an error it can
// see. That is the deliberate trade: an explicit failure over a silent swap.
func (c *Chain) pinnedProvider(providerName, model, requested string) (brain.Provider, error) {
	if providerName == "" {
		if c.retired(requested) {
			// The name is one this deployment used to publish and never will
			// again, so "retry shortly" would be a lie with a deadline that
			// never arrives. The remedy is a re-fetch, not a retry, and the
			// message says so — the caller cannot derive the new identifier,
			// only the listing can hand it over.
			return nil, fmt.Errorf("%w: %q was published before the serving host was renamed and is no longer offered; re-fetch the current identifiers from /v1/models",
				ErrRetiredIdentifier, requested)
		}
		return nil, fmt.Errorf("%w: model %q (requested as %q) is not served by any registered provider",
			ErrPinnedModelUnavailable, model, requested)
	}
	provider, ok := c.providers[providerName]
	if !ok {
		// Not an availability condition: the Brain and the chain disagree about
		// which providers exist, which is a wiring fault in this build and
		// stays a 500. Retrying cannot fix it.
		return nil, fmt.Errorf(
			"provider %q serves model %q (requested as %q) but is not registered with the chain",
			providerName, model, requested)
	}
	if !provider.Available() {
		return nil, fmt.Errorf("%w: provider %q serving model %q (requested as %q) is not available",
			ErrPinnedModelUnavailable, providerName, model, requested)
	}
	return provider, nil
}

// Entries returns a snapshot copy of the current entry list.
// It is safe to call concurrently.
func (c *Chain) Entries() []ChainEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]ChainEntry, len(c.entries))
	copy(out, c.entries)
	return out
}

// Complete iterates the entry list in order and calls the first available
// provider.  On success it returns the response.  On 429 it marks the entry
// exhausted and tries the next one.  On any other error it records a circuit
// failure and tries the next one.
//
// If every entry is skipped or fails, Complete returns an error that wraps
// the last provider error and contains the string "all providers exhausted".
//
// State mutations (429 cooldowns, circuit-breaker failures) are written back
// directly into c.entries under a write lock so the changes persist across
// successive calls.  The old snapshot-based approach (Entries() returning a
// copy) silently discarded every mutation.
func (c *Chain) Complete(ctx context.Context, req *types.InternalChatRequest) (*types.InternalChatResponse, error) {
	// A request that names a served model is answered by THAT model. This runs
	// before the entry list because the loop below overrides req.Model with its
	// own entry default — resolving afterwards would be resolving something the
	// chain had already thrown away.
	if providerName, model, pinned := c.pin(req.Model); pinned {
		provider, err := c.pinnedProvider(providerName, model, req.Model)
		if err != nil {
			return nil, err
		}
		reqCopy := deepCopyRequest(req)
		reqCopy.Model = model
		// The tool-capability skip below is not applied here: it exists to move
		// a tool-bearing request to a DIFFERENT entry, and a pinned request has
		// nowhere else to go. If the pinned model rejects tools, the provider's
		// own error says so — which is the truth the caller needs.
		return provider.Complete(ctx, &reqCopy)
	}

	// The pin missed. A caller that named a model this deployment does not
	// serve is refused here, before the entry list below overrides req.Model
	// with its own default and answers from a model the caller never asked
	// for. See refuseUnknownModel.
	if err := c.refuseUnknownModel(req.Model); err != nil {
		return nil, err
	}

	c.mu.RLock()
	numEntries := len(c.entries)
	c.mu.RUnlock()

	var lastErr error
	for i := 0; i < numEntries; i++ {
		// Read entry state under a read lock (copy for local use).
		c.mu.RLock()
		if i >= len(c.entries) {
			c.mu.RUnlock()
			break
		}
		entry := c.entries[i] // value copy for reading
		c.mu.RUnlock()

		if !c.entryAvailable(i) {
			slog.Debug("fallback: skipping unavailable entry", "provider", entry.ProviderName)
			continue
		}

		// Skip rate-limited cloud providers; local fallbacks always get a shot.
		if c.rateLimiter.ShouldSkip(entry.ProviderName) && !entry.IsLocalFallback {
			slog.Debug("fallback: skipping rate-limited entry", "provider", entry.ProviderName)
			continue
		}

		provider, ok := c.providers[entry.ProviderName]
		if !ok {
			slog.Warn("fallback: provider not registered", "provider", entry.ProviderName)
			continue
		}

		// Skip entries whose model is not known to support tool calling when the
		// request carries tools.  This avoids sending tool-bearing requests to
		// models such as Gemma or GLM-5.1 that reject the tools field entirely.
		// The entry is not marked Exhausted — it may still serve tool-free requests.
		modelID := entry.ModelID
		if modelID == "" {
			modelID = req.Model
		}
		if len(req.Tools) > 0 && !modelSupportsTools(modelID) {
			slog.Debug("fallback: skipping non-tool-capable model for tool request",
				"provider", entry.ProviderName, "model", modelID)
			continue
		}

		// Build a per-entry request: deep copy and override Model if specified.
		reqCopy := deepCopyRequest(req)
		if entry.ModelID != "" {
			reqCopy.Model = entry.ModelID
		}

		resp, err := provider.Complete(ctx, &reqCopy)
		if err == nil {
			c.writeBackSuccess(i)
			c.rateLimiter.ResetBackoff(entry.ProviderName)
			return resp, nil
		}

		// Record the failure and decide on the next action.
		lastErr = err
		c.writeBackError(i, err)
		slog.Warn("fallback: provider error, trying next",
			"provider", entry.ProviderName,
			"error", err)
	}

	if lastErr != nil {
		return nil, fmt.Errorf("%w, last error: %w", ErrProvidersExhausted, lastErr)
	}
	return nil, fmt.Errorf("%w: no entries available", ErrProvidersExhausted)
}

// CompleteStream iterates the entry list in order and calls the first
// available provider for streaming.  The failover logic mirrors Complete.
func (c *Chain) CompleteStream(ctx context.Context, req *types.InternalChatRequest) (<-chan types.StreamChunk, error) {
	// See Complete: the pin is resolved before the entry list, on its own call
	// site. Streaming is how a chat client actually talks, so an unguarded
	// streaming path would leave the defect live for the common case.
	if providerName, model, pinned := c.pin(req.Model); pinned {
		provider, err := c.pinnedProvider(providerName, model, req.Model)
		if err != nil {
			return nil, err
		}
		reqCopy := deepCopyRequest(req)
		reqCopy.Model = model
		return provider.CompleteStream(ctx, &reqCopy)
	}

	// Same refusal as Complete, on its own call site: streaming is how a chat
	// client actually talks, so a guard that covered only the buffered path
	// would leave the misroute live for the common case.
	if err := c.refuseUnknownModel(req.Model); err != nil {
		return nil, err
	}

	c.mu.RLock()
	numEntries := len(c.entries)
	c.mu.RUnlock()

	var lastErr error
	for i := 0; i < numEntries; i++ {
		// Read entry state under a read lock (copy for local use).
		c.mu.RLock()
		if i >= len(c.entries) {
			c.mu.RUnlock()
			break
		}
		entry := c.entries[i] // value copy for reading
		c.mu.RUnlock()

		if !c.entryAvailable(i) {
			slog.Debug("fallback: skipping unavailable entry (stream)", "provider", entry.ProviderName)
			continue
		}

		if c.rateLimiter.ShouldSkip(entry.ProviderName) && !entry.IsLocalFallback {
			slog.Debug("fallback: skipping rate-limited entry (stream)", "provider", entry.ProviderName)
			continue
		}

		provider, ok := c.providers[entry.ProviderName]
		if !ok {
			slog.Warn("fallback: provider not registered (stream)", "provider", entry.ProviderName)
			continue
		}

		// Skip entries whose model is not known to support tool calling when the
		// request carries tools (mirrors the non-streaming path).
		streamModelID := entry.ModelID
		if streamModelID == "" {
			streamModelID = req.Model
		}
		if len(req.Tools) > 0 && !modelSupportsTools(streamModelID) {
			slog.Debug("fallback: skipping non-tool-capable model for tool request (stream)",
				"provider", entry.ProviderName, "model", streamModelID)
			continue
		}

		reqCopy := deepCopyRequest(req)
		if entry.ModelID != "" {
			reqCopy.Model = entry.ModelID
		}

		ch, err := provider.CompleteStream(ctx, &reqCopy)
		if err == nil {
			c.writeBackSuccess(i)
			c.rateLimiter.ResetBackoff(entry.ProviderName)
			return ch, nil
		}

		lastErr = err
		c.writeBackError(i, err)
		slog.Warn("fallback: provider stream error, trying next",
			"provider", entry.ProviderName,
			"error", err)
	}

	if lastErr != nil {
		return nil, fmt.Errorf("%w, last error: %w", ErrProvidersExhausted, lastErr)
	}
	return nil, fmt.Errorf("%w: no entries available", ErrProvidersExhausted)
}

// entryAvailable checks whether c.entries[idx] can accept a request, also
// performing the auto-recovery from Exhausted status when the cooldown has
// elapsed.  Mutations (status reset) are written back under a write lock so
// they persist across calls.
func (c *Chain) entryAvailable(idx int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if idx >= len(c.entries) {
		return false
	}
	e := &c.entries[idx]

	// Auto-recover from exhaustion once the cooldown window has elapsed.
	if e.Status == EntryExhausted {
		if !e.CooldownUntil.IsZero() && time.Now().After(e.CooldownUntil) {
			e.Status = EntryActive
			e.CooldownUntil = time.Time{}
		} else {
			return false
		}
	}

	// Defer to the circuit breaker if one is attached.
	if e.CircuitBreaker != nil && !e.CircuitBreaker.Allow() {
		return false
	}

	return true
}

// writeBackError inspects err and mutates c.entries[idx] directly under a
// write lock so the changes are visible to subsequent calls.
//
//   - HTTP 429: mark the entry Exhausted with a CooldownUntil derived from the
//     Retry-After header or the exponential-backoff sequence.
//   - HTTP 5xx: record a circuit-breaker failure.
//   - Any other error: record a circuit-breaker failure.
func (c *Chain) writeBackError(idx int, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if idx >= len(c.entries) {
		return
	}
	entry := &c.entries[idx]

	var pe *brain.ProviderError
	if brain.AsProviderError(err, &pe) {
		switch {
		case pe.StatusCode == 429:
			backoff := c.rateLimiter.ParseRetryAfter(pe.Headers)
			if backoff <= 0 {
				backoff = c.rateLimiter.NextBackoff(entry.ProviderName)
			}
			entry.Status = EntryExhausted
			entry.CooldownUntil = time.Now().Add(backoff)
			slog.Warn("fallback: rate-limited, cooling down",
				"provider", entry.ProviderName,
				"cooldown", backoff)

		case pe.StatusCode >= 500:
			if entry.CircuitBreaker != nil {
				entry.CircuitBreaker.RecordFailure()
			}
			slog.Warn("fallback: upstream 5xx, recording circuit failure",
				"provider", entry.ProviderName,
				"status", pe.StatusCode)

		default:
			// 4xx other than 429: treat as a transient circuit failure.
			if entry.CircuitBreaker != nil {
				entry.CircuitBreaker.RecordFailure()
			}
		}
		return
	}

	// Non-ProviderError (network error, context cancelled, etc.).
	if entry.CircuitBreaker != nil {
		entry.CircuitBreaker.RecordFailure()
	}
}

// writeBackSuccess records a success on the circuit breaker for c.entries[idx]
// under a write lock.
func (c *Chain) writeBackSuccess(idx int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if idx >= len(c.entries) {
		return
	}
	if cb := c.entries[idx].CircuitBreaker; cb != nil {
		cb.RecordSuccess()
	}
}

// deepCopyRequest returns a shallow-struct copy of req with the slice fields
// (Messages, Tools) duplicated so that providers cannot mutate each other's
// view of the request when the chain retries on the next entry.
func deepCopyRequest(req *types.InternalChatRequest) types.InternalChatRequest {
	cp := *req
	if len(req.Messages) > 0 {
		cp.Messages = make([]types.InternalMessage, len(req.Messages))
		copy(cp.Messages, req.Messages)
	}
	if len(req.Tools) > 0 {
		cp.Tools = make([]types.InternalTool, len(req.Tools))
		copy(cp.Tools, req.Tools)
	}
	return cp
}
