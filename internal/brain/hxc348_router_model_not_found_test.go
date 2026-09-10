// HXC-348, router half — Route must refuse a named model nothing serves.
//
// # Why this exists alongside the gateway guard
//
// The shipped binary wires a *fallback.Chain as the gateway's Completer, so
// the HTTP-level guard in internal/gateway is the one that pins production
// behaviour. But Router.Route is a SECOND, independently reachable copy of
// the same fail-open: steps 4 (fallback provider) and 5 (any available
// provider) run for a request that named a model no provider serves, so a
// *brain.Brain used directly as the Completer — which internal/gateway's
// router.go explicitly still allows — substitutes a different model with no
// error. Fixing one and not the other leaves the defect live behind a wiring
// switch.
//
// # The boundary this draws, deliberately
//
// "Unresolved" here means NO registered provider claims the name — not "the
// provider that claims it is down". A model some provider serves whose
// backend is unavailable is an AVAILABILITY condition, which this project
// already answers with the retryable status via fallback.ErrPinnedModelUnavailable,
// and TestRouter_FallbackWhenPreferredUnavailable pins the existing routing
// behaviour for it. Refusing that case here would change an unrelated answer
// under cover of this fix, so the guard consults the provider's model list
// WITHOUT consulting its health.
package brain_test

import (
	"testing"

	"github.com/HelixDevelopment/HelixLLM/internal/brain"
	"github.com/HelixDevelopment/HelixLLM/pkg/types"
)

// A named model nothing serves must produce an identifiable refusal rather
// than a confident answer from an unrelated provider.
func TestHXC348_Route_UnknownModelIsRefused(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	r.Register("llamacpp", newMock("llamacpp", true, "qwen2.5-coder-3b"))

	p, err := r.Route(&types.InternalChatRequest{Model: "definitely-not-a-real-model-zzz"})
	if err == nil {
		t.Fatalf("Route returned provider %q for a model no provider serves; "+
			"the caller named one model and would have been answered by another",
			p.Name())
	}
	if !brain.IsModelNotFound(err) {
		t.Errorf("IsModelNotFound(%v) = false; the HTTP boundary has to tell "+
			"\"this deployment does not serve that name\" (404, never retry) from "+
			"\"nothing is up right now\" (503, retry) without matching message text", err)
	}
	if name, ok := brain.NotFoundModelName(err); !ok || name != "definitely-not-a-real-model-zzz" {
		t.Errorf("NotFoundModelName(%v) = (%q, %v); the error must carry the requested "+
			"name so the response can tell the caller which model was refused",
			err, name, ok)
	}
}

// The legitimate fallback: a caller that names NO model has expressed no
// preference, and the score-ordered/default choice is the feature.
func TestHXC348_Route_EmptyModelStillFallsBack(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	r.Register("llamacpp", newMock("llamacpp", true, "qwen2.5-coder-3b"))

	p, err := r.Route(&types.InternalChatRequest{Model: ""})
	if err != nil {
		t.Fatalf("Route(no model) returned error %v; a caller that named nothing "+
			"must still be served", err)
	}
	if p.Name() != "llamacpp" {
		t.Errorf("Route(no model) provider = %q, want %q", p.Name(), "llamacpp")
	}
}

// Negative control for the boundary above: a model a registered provider
// DOES claim, whose provider is merely down, is not a not-found condition and
// keeps its existing routing.
func TestHXC348_Route_KnownModelOnDownProviderIsNotNotFound(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	r.Register("openai", newMock("openai", false, "gpt-4o"))
	r.Register("llamacpp", newMock("llamacpp", true, "llama-3.1-70b"))

	p, err := r.Route(&types.InternalChatRequest{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Route returned error %v for a model a registered provider serves; "+
			"a provider being down is an availability condition, not an unknown name", err)
	}
	if p.Name() != "llamacpp" {
		t.Errorf("Route() provider = %q, want %q", p.Name(), "llamacpp")
	}
}

// The served model must still route. Without this, a guard that refused
// everything would satisfy the case above.
func TestHXC348_Route_ServedModelStillRoutes(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	r.Register("llamacpp", newMock("llamacpp", true, "qwen2.5-coder-3b"))

	p, err := r.Route(&types.InternalChatRequest{Model: "qwen2.5-coder-3b"})
	if err != nil {
		t.Fatalf("Route returned error for a served model: %v", err)
	}
	if p.Name() != "llamacpp" {
		t.Errorf("Route() provider = %q, want %q", p.Name(), "llamacpp")
	}
}

// An exhausted deployment reports exhaustion, not not-found: the name may be
// perfectly good, and telling the caller it does not exist would send it to
// change a model id that was never the problem.
func TestHXC348_Route_NoProvidersAvailableIsNotNotFound(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	r.Register("llamacpp", newMock("llamacpp", false, "qwen2.5-coder-3b"))

	_, err := r.Route(&types.InternalChatRequest{Model: "qwen2.5-coder-3b"})
	if err == nil {
		t.Fatal("expected an error when no provider is available")
	}
	if brain.IsModelNotFound(err) {
		t.Errorf("IsModelNotFound(%v) = true for a model this deployment DOES serve; "+
			"a down backend is retryable and must not be reported as a name that "+
			"does not exist", err)
	}
}
