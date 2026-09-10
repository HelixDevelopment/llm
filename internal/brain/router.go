package brain

import (
	"fmt"
	"strings"

	"github.com/HelixDevelopment/HelixLLM/pkg/types"
)

// RoutingRule maps a model-name prefix to a named provider.
type RoutingRule struct {
	Prefix   string // model name prefix, e.g. "gpt-", "claude-", "llama-"
	Provider string // provider name as registered with Register
}

// Router selects the right Provider for each InternalChatRequest.
//
// Selection priority:
//  1. req.Provider explicit override (if that provider is registered and available)
//  2. Exact model match against registered providers' model lists (if available)
//  3. Prefix-match rules built dynamically from provider models (if available)
//  4. Refuse a NAMED model no registered provider claims (ErrModelNotFound)
//  5. Fallback provider (if available)
//  6. Any available provider
//  7. Error
//
// Steps 5 and 6 are the substitution the router exists to provide, and step 4
// is the boundary that keeps them honest: they answer a caller who named NO
// model, never one who named a model this deployment has never heard of. See
// Route for why that distinction is the whole point.
type Router struct {
	providers map[string]Provider
	rules     []RoutingRule
	fallback  string
}

// NewRouter creates a Router with the given fallback provider name. Routing
// rules are built dynamically from registered providers' model lists — there
// are no hardcoded prefix mappings.
func NewRouter(fallback string) *Router {
	return &Router{
		providers: make(map[string]Provider),
		fallback:  fallback,
	}
}

// Register adds a provider under the given name. Subsequent calls with the
// same name overwrite the previous registration. After every registration the
// routing rules are rebuilt from all providers' model lists.
func (r *Router) Register(name string, p Provider) {
	r.providers[name] = p
	r.rebuildRules()
}

// rebuildRules derives prefix-based routing rules from the model IDs reported
// by every registered provider. For each model, the longest dash-delimited
// prefix (e.g. "gpt-" from "gpt-4o", "deepseek-" from "deepseek-chat") is
// used as a routing prefix. Exact-match routing is handled directly in Route
// without rules.
func (r *Router) rebuildRules() {
	seen := make(map[string]string) // prefix → provider name (first wins)
	var rules []RoutingRule

	for name, p := range r.providers {
		for _, model := range p.Models() {
			// Build a prefix from the model ID. Use everything up to and
			// including the first '-'. For IDs without a dash the whole
			// string is used as a prefix so exact matches still work.
			prefix := model
			if idx := strings.Index(model, "-"); idx >= 0 {
				prefix = model[:idx+1]
			}
			if _, ok := seen[prefix]; !ok {
				seen[prefix] = name
				rules = append(rules, RoutingRule{
					Prefix:   prefix,
					Provider: name,
				})
			}
		}
	}

	r.rules = rules
}

// Route selects a Provider for the given request following the priority rules
// described on Router.
func (r *Router) Route(req *types.InternalChatRequest) (Provider, error) {
	// 1. Explicit provider override.
	if req.Provider != "" {
		if p, ok := r.providers[string(req.Provider)]; ok && p.Available() {
			return p, nil
		}
	}

	// claimed records that SOME registered provider offers this name, even if
	// the provider that offers it turned out to be down. It is what separates
	// the two reasons steps 2 and 3 can fail — see step 4.
	claimed := false

	// 2. Exact model match — check every provider's model list.
	for _, p := range r.providers {
		for _, m := range p.Models() {
			if m != req.Model {
				continue
			}
			if p.Available() {
				return p, nil
			}
			claimed = true
		}
	}

	// 3. Prefix-match rules (built dynamically from provider models).
	for _, rule := range r.rules {
		if !strings.HasPrefix(req.Model, rule.Prefix) {
			continue
		}
		if p, ok := r.providers[rule.Provider]; ok {
			if p.Available() {
				return p, nil
			}
			claimed = true
		}
	}

	// 4. A model this deployment does not serve is REFUSED, not substituted.
	//
	// Steps 5 and 6 below hand back a provider chosen without reference to
	// req.Model. For a request that named no model that is the intended
	// behaviour — the caller expressed no preference and any available
	// backend answers it. For a request that named one, it is a silent
	// misroute: the caller gets a confident completion from a model it never
	// asked for, with a 200 and no way to detect the substitution. Measured
	// on the shipped gateway, `definitely-not-a-real-model-zzz`, `gpt-4o` and
	// `claude-opus-4` were all answered by the one local model, and the
	// naming layer's own comment (see ResolveModelName) already called this
	// out as "a silent misroute" without guarding it here.
	//
	// The condition is narrow on purpose. `claimed` is set when some provider
	// LISTS the name and was skipped only for being unavailable, which is an
	// availability condition the deployment reports elsewhere as retryable —
	// refusing it here would change an unrelated answer under cover of this
	// fix, and TestRouter_FallbackWhenPreferredUnavailable pins that
	// behaviour. Availability is therefore never consulted to decide
	// not-found; only the model lists are.
	// A DELEGATION SENTINEL is exempt for the same reason the empty model is:
	// it names no model to be missing. `auto` is the documented "you choose"
	// value, so a registry miss on it carries no information and refusing it
	// would 404 the project's own canonical first API call. The predicate is
	// shared with fallback.Chain.refuseUnknownModel — see delegation_sentinel.go
	// for why one definition rather than two.
	if !IsDelegationSentinel(req.Model) && !claimed {
		return nil, NewModelNotFound(req.Model)
	}

	// 5. Fallback provider.
	if r.fallback != "" {
		if p, ok := r.providers[r.fallback]; ok && p.Available() {
			return p, nil
		}
	}

	// 6. Any available provider.
	for _, p := range r.providers {
		if p.Available() {
			return p, nil
		}
	}

	return nil, fmt.Errorf("router: no available provider for model %q", req.Model)
}
