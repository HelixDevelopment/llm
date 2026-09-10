package brain_test

import (
	"testing"

	"github.com/HelixDevelopment/HelixLLM/internal/brain"
	"github.com/HelixDevelopment/HelixLLM/pkg/types"
)

// newMock is a convenience helper for router tests.
func newMock(name string, available bool, models ...string) *mockProvider {
	return &mockProvider{
		name:      name,
		available: available,
		models:    models,
	}
}

func TestRouter_RouteByModelPrefix_GPT(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	r.Register("openai", newMock("openai", true, "gpt-4o"))
	r.Register("anthropic", newMock("anthropic", true, "claude-sonnet-4-5"))
	r.Register("llamacpp", newMock("llamacpp", true, "llama-3.1-70b"))

	p, err := r.Route(&types.InternalChatRequest{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}
	if p.Name() != "openai" {
		t.Errorf("Route() provider = %q, want %q", p.Name(), "openai")
	}
}

func TestRouter_RouteByModelPrefix_Claude(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	r.Register("openai", newMock("openai", true, "gpt-4o"))
	r.Register("anthropic", newMock("anthropic", true, "claude-sonnet-4-5"))
	r.Register("llamacpp", newMock("llamacpp", true, "llama-3.1-70b"))

	p, err := r.Route(&types.InternalChatRequest{Model: "claude-sonnet-4-5"})
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}
	if p.Name() != "anthropic" {
		t.Errorf("Route() provider = %q, want %q", p.Name(), "anthropic")
	}
}

func TestRouter_RouteByModelPrefix_Llama(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	r.Register("openai", newMock("openai", true, "gpt-4o"))
	r.Register("anthropic", newMock("anthropic", true, "claude-sonnet-4-5"))
	r.Register("llamacpp", newMock("llamacpp", true, "llama-3.1-70b"))

	p, err := r.Route(&types.InternalChatRequest{Model: "llama-3.1-70b"})
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}
	if p.Name() != "llamacpp" {
		t.Errorf("Route() provider = %q, want %q", p.Name(), "llamacpp")
	}
}

func TestRouter_RouteByModelPrefix_Qwen(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	r.Register("llamacpp", newMock("llamacpp", true, "qwen2.5-72b"))

	p, err := r.Route(&types.InternalChatRequest{Model: "qwen2.5-72b"})
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}
	if p.Name() != "llamacpp" {
		t.Errorf("Route() provider = %q, want %q", p.Name(), "llamacpp")
	}
}

func TestRouter_ExplicitProviderOverride(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	r.Register("openai", newMock("openai", true, "gpt-4o"))
	r.Register("anthropic", newMock("anthropic", true, "claude-sonnet-4-5"))
	r.Register("llamacpp", newMock("llamacpp", true, "llama-3.1-70b"))

	// Request a gpt model but force anthropic via explicit provider field.
	p, err := r.Route(&types.InternalChatRequest{
		Model:    "gpt-4o",
		Provider: types.ProviderAnthropic,
	})
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}
	if p.Name() != "anthropic" {
		t.Errorf("Route() provider = %q, want %q", p.Name(), "anthropic")
	}
}

func TestRouter_FallbackWhenPreferredUnavailable(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	// openai is unavailable; llamacpp is the fallback and is available.
	r.Register("openai", newMock("openai", false, "gpt-4o"))
	r.Register("llamacpp", newMock("llamacpp", true, "llama-3.1-70b"))

	p, err := r.Route(&types.InternalChatRequest{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}
	if p.Name() != "llamacpp" {
		t.Errorf("Route() provider = %q, want %q", p.Name(), "llamacpp")
	}
}

// TestRouter_FallbackToAnyAvailable pins step 6: with no fallback provider
// name configured, the only available provider answers.
//
// # Reconciled for HXC-348 (§11.4.120)
//
// This case used to drive step 6 with the model name "unknown-model" and
// assert that the request was answered anyway. That assertion WAS the
// fail-open HXC-348 fixes: a caller naming a model no provider serves was
// answered, at 200, by an unrelated one. The fix makes that an error, so this
// case had to be reconciled rather than left red — and reconciled by
// asserting the NEW mechanism, not by weakening the assertion.
//
// The behaviour it legitimately pinned — "no fallback name configured, so any
// available provider answers" — is unchanged and still pinned here; it is
// reached the way a caller with no preference actually reaches it, by naming
// no model at all. The half that asserted the defect moved to
// TestRouter_UnknownNamedModelIsRefused below, with its polarity flipped.
func TestRouter_FallbackToAnyAvailable(t *testing.T) {
	// No fallback name set; the only available provider should be chosen.
	r := brain.NewRouter("")
	r.Register("openai", newMock("openai", false, "gpt-4o"))
	r.Register("anthropic", newMock("anthropic", true, "claude-sonnet-4-5"))

	// No model named: the caller expressed no preference, which is the
	// request this substitution exists to serve.
	p, err := r.Route(&types.InternalChatRequest{})
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}
	if p.Name() != "anthropic" {
		t.Errorf("Route() provider = %q, want %q", p.Name(), "anthropic")
	}
}

// TestRouter_UnknownNamedModelIsRefused is the other half of the case above:
// the same router, the same providers, but a request that NAMES a model no
// provider serves. It must be refused rather than answered by whichever
// provider happens to be up.
func TestRouter_UnknownNamedModelIsRefused(t *testing.T) {
	r := brain.NewRouter("")
	r.Register("openai", newMock("openai", false, "gpt-4o"))
	r.Register("anthropic", newMock("anthropic", true, "claude-sonnet-4-5"))

	p, err := r.Route(&types.InternalChatRequest{Model: "unknown-model"})
	if err == nil {
		t.Fatalf("Route() returned provider %q for a model no provider serves", p.Name())
	}
	if !brain.IsModelNotFound(err) {
		t.Errorf("IsModelNotFound(%v) = false, want true", err)
	}
}

func TestRouter_ErrorWhenNoProvidersAvailable(t *testing.T) {
	r := brain.NewRouter("openai")
	r.Register("openai", newMock("openai", false))
	r.Register("anthropic", newMock("anthropic", false))

	_, err := r.Route(&types.InternalChatRequest{Model: "gpt-4o"})
	if err == nil {
		t.Fatal("expected error when no providers available, got nil")
	}
}

func TestRouter_ErrorWhenNoProvidersRegistered(t *testing.T) {
	r := brain.NewRouter("openai")

	_, err := r.Route(&types.InternalChatRequest{Model: "gpt-4o"})
	if err == nil {
		t.Fatal("expected error when no providers registered, got nil")
	}
}

func TestRouter_ExplicitProviderUnavailableFallsThrough(t *testing.T) {
	r := brain.NewRouter("llamacpp")
	// Explicit provider is registered but unavailable; should fall through to prefix rule.
	r.Register("openai", newMock("openai", false, "gpt-4o"))
	r.Register("llamacpp", newMock("llamacpp", true, "llama-3.1-70b"))

	p, err := r.Route(&types.InternalChatRequest{
		Model:    "gpt-4o",
		Provider: types.ProviderOpenAI, // openai unavailable → falls through
	})
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}
	// gpt- prefix matches openai (unavailable), then fallback llamacpp is used.
	if p.Name() != "llamacpp" {
		t.Errorf("Route() provider = %q, want %q", p.Name(), "llamacpp")
	}
}
