// HXC-348 follow-up — a DELEGATION SENTINEL must fall through, not be refused.
//
// # The regression this reproduces
//
// The HXC-348 fix refuses a model name the deployment does not serve. It
// derived "does not serve" from ModelPinner.PinModel returning ok=false, and
// exempted exactly one input: the empty string. That is one exemption short.
//
// `auto` is a documented sentinel meaning "you choose" — the semantic OPPOSITE
// of "I claim a specific model you must have". It is the value the project's
// own canonical first API call uses:
//
//	website/content/_index.md:29                              {"model":"auto"}
//	docs/courses/01-getting-started/lesson-01-introduction.md:131
//	docs/courses/01-getting-started/lesson-03-first-api-call.md:202
//	challenges/banks/performance/nonblocking.yaml:34,65
//	tests/e2e/e2e_test.go:60, tests/e2e/multi_provider_e2e_test.go:24
//	tests/stress/stress_test.go:34, tests/integration/auth_test.go:51
//
// No production code special-cased the literal, so pre-fix it fell through
// PinModel's ok=false into the score-ordered chain and was answered. Post-fix
// it was refused as an unknown model, and TestAuth_ChatWithoutAuth started
// getting 404 where it wants 200-or-503 — a documented client flow broken by
// a fix aimed at undocumented ones.
//
// # Why this is not a weakening of HXC-348
//
// HXC-348 exists to stop the gateway answering "I want gpt-4o" with a
// different model behind a 200. A sentinel makes no such claim: it delegates
// the choice, so answering it from the chain is obedience, not substitution.
// The refusal that matters — a NAMED, SPECIFIC, unserved model — is asserted
// unchanged here (see the "still refused" cases), so a change that let
// everything through fails this file rather than passing it.
//
// # Coverage gap this closes
//
// Neither hxc348_model_not_found_test.go nor hxc348_router_model_not_found_test.go
// exercised `auto` at all, on any path. Only the pre-existing auth test caught
// the regression, and only on the buffered path. Every sentinel is therefore
// driven here across ALL THREE serving paths — buffered, streaming, and
// /v1/completions — because those are three separate call sites into the
// chain's pin, and a guard on one says nothing about the other two.
//
// # Polarity switch (§11.4.115)
//
// RED_MODE=1 reproduces the REGRESSION on the artifact that carries it: it
// asserts a sentinel IS refused with 404, so a run against the unfixed tree
// PASSES and proves the reproduction is real rather than synthetic.
// RED_MODE=0 (the default) is the standing GREEN guard: the sentinel is
// honoured and answered.
//
//	RED_MODE=1 go test -run TestHXC348Sentinel ./internal/gateway/   # pre-fix
//	           go test -run TestHXC348Sentinel ./internal/gateway/   # post-fix
package gateway_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/HelixDevelopment/HelixLLM/internal/gateway"
	"github.com/HelixDevelopment/HelixLLM/pkg/api"
)

// hxc348Sentinels is every delegation sentinel found in documented use, plus
// the case renderings a real caller sends. The audit that produced the base
// set is recorded in docs/qa/hxc348_auto_sentinel_20260908T203517Z/.
//
// A sentinel is a KEYWORD, not a model id, so it is matched case-folded: a
// caller who sends "Auto" has expressed the same delegation as one who sent
// "auto", and answering one but 404-ing the other would be a distinction with
// no meaning behind it.
//
// A SPACE-PADDED rendering ("  auto  ") is deliberately absent: the gateway's
// model-name charset validation rejects it with 400 before the chain is
// reached at all. That is pre-existing behaviour unrelated to HXC-348, and
// TestHXC348Sentinel_SpacePaddedSentinelIsRejectedAtValidation pins it so the
// boundary is recorded rather than assumed.
var hxc348Sentinels = []string{
	"auto",
	"AUTO",
	"Auto",
	"default",
	"DEFAULT",
}

// hxc348SentinelLookalikes are names that CONTAIN a sentinel but are not one.
// They are the negative control for the matcher: a substring or prefix match
// would let these through, silently re-opening HXC-348 for every id that
// happens to start with "auto". They MUST still be refused.
var hxc348SentinelLookalikes = []string{
	"auto-gpt",
	"automatic",
	"autopilot-7b",
	"default-model",
	"gpt-auto",
	"auto/llama",
}

// hxc348AssertSentinelHonoured applies the polarity switch for a sentinel.
//
// GREEN: the request is answered (200) — the sentinel delegated the choice and
// the chain made it.
// RED: the regressed artifact refuses it with 404 model_not_found, which is
// the defect, observed rather than assumed.
func hxc348AssertSentinelHonoured(t *testing.T, route, sentinel string, w *httptest.ResponseRecorder) {
	t.Helper()

	if redMode() {
		if w.Code != http.StatusNotFound {
			t.Errorf("%s model=%q: RED_MODE expects the regression (404), got status %d body %s",
				route, sentinel, w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "model_not_found") {
			t.Errorf("%s model=%q: RED_MODE expects a model_not_found refusal, got %s",
				route, sentinel, w.Body.String())
		}
		return
	}

	if w.Code != http.StatusOK {
		t.Errorf("%s model=%q: status = %d, want %d — a delegation sentinel means "+
			"\"you choose\", so the chain must answer it exactly as it answers an "+
			"empty model; refusing it breaks a documented client flow; body: %s",
			route, sentinel, w.Code, http.StatusOK, w.Body.String())
		return
	}
	if strings.Contains(w.Body.String(), "model_not_found") {
		t.Errorf("%s model=%q: answered 200 but the body still carries a "+
			"model_not_found refusal: %s", route, sentinel, w.Body.String())
	}
}

// ── The three serving paths, every sentinel ──────────────────────────────

func TestHXC348Sentinel_ChatCompletions_SentinelIsHonoured(t *testing.T) {
	for _, sentinel := range hxc348Sentinels {
		t.Run(sentinel, func(t *testing.T) {
			hxc348AssertSentinelHonoured(t, "POST /v1/chat/completions", sentinel,
				hxc348Post(t,
					gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
					"/v1/chat/completions",
					api.ChatCompletionRequest{
						Model:    sentinel,
						Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
					}))
		})
	}
}

// Streaming is a SEPARATE call site into the chain's pin (Chain.CompleteStream
// carries its own refusal), and it is how a chat client actually talks. A
// sentinel fixed only on the buffered path would leave the regression live for
// the common case — which is exactly the shape of the gap that let the
// original regression ship.
func TestHXC348Sentinel_ChatCompletionsStreaming_SentinelIsHonoured(t *testing.T) {
	for _, sentinel := range hxc348Sentinels {
		t.Run(sentinel, func(t *testing.T) {
			hxc348AssertSentinelHonoured(t, "POST /v1/chat/completions (stream)", sentinel,
				hxc348Post(t,
					gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
					"/v1/chat/completions",
					api.ChatCompletionRequest{
						Model:    sentinel,
						Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
						Stream:   true,
					}))
		})
	}
}

// /v1/completions reaches the chain through HandleCompletions' own defaulting,
// which the HXC-348 fix changed to pass req.Model through untouched. That makes
// it a third distinct path to the same refusal, so it gets its own coverage.
func TestHXC348Sentinel_Completions_SentinelIsHonoured(t *testing.T) {
	for _, sentinel := range hxc348Sentinels {
		t.Run(sentinel, func(t *testing.T) {
			hxc348AssertSentinelHonoured(t, "POST /v1/completions", sentinel,
				hxc348Post(t,
					gateway.HandleCompletions(hxc348Chain(t)),
					"/v1/completions",
					map[string]any{"model": sentinel, "prompt": "Hello"}))
		})
	}
}

// A streaming sentinel must produce a real SSE chunk stream, not merely a 200.
// Without this, a handler that returned an empty 200 body would satisfy every
// status assertion above while delivering nothing to the client.
func TestHXC348Sentinel_ChatCompletionsStreaming_SentinelStreamsRealChunks(t *testing.T) {
	if redMode() {
		t.Skip("RED_MODE: the regressed artifact refuses the sentinel, so there is no stream to inspect")
	}
	w := hxc348Post(t,
		gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
		"/v1/chat/completions",
		api.ChatCompletionRequest{
			Model:    "auto",
			Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
			Stream:   true,
		})
	if w.Code != http.StatusOK {
		t.Fatalf("sentinel (stream): status = %d, want 200; body: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "chat.completion.chunk") {
		t.Errorf("sentinel (stream): body is not an SSE chunk stream: %s", w.Body.String())
	}
}

// A sentinel delegates the choice, so the answer must come from a model the
// deployment actually serves — the same model an empty request gets. This is
// what distinguishes "honoured" from "echoed the sentinel back as if it were a
// model id", which would satisfy a status-only assertion while telling the
// client it was answered by a model called "auto".
func TestHXC348Sentinel_ChatCompletions_SentinelIsAnsweredByAServedModel(t *testing.T) {
	if redMode() {
		t.Skip("RED_MODE: the regressed artifact refuses the sentinel, so there is no response to inspect")
	}
	w := hxc348Post(t,
		gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
		"/v1/chat/completions",
		api.ChatCompletionRequest{
			Model:    "auto",
			Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
		})
	if w.Code != http.StatusOK {
		t.Fatalf("sentinel: status = %d, want 200; body: %s", w.Code, w.Body.String())
	}
	var resp api.ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("sentinel: body is not a completion response: %v (%s)", err, w.Body.String())
	}
	if resp.Model != hxc348ServedModel {
		t.Errorf("sentinel: answered by %q, want the served model %q — a sentinel "+
			"delegates the choice, so the response must name the model that actually "+
			"answered, never the sentinel itself", resp.Model, hxc348ServedModel)
	}
}

// ── Negative control: HXC-348's refusal is NOT weakened ──────────────────

// The whole point of HXC-348 is that a NAMED, SPECIFIC, unserved model is
// refused. A sentinel matcher that matched on prefix or substring would let
// every one of these through and silently re-open the defect. They must still
// be refused in BOTH polarities, which is why this case does not consult
// redMode: the pre-fix and post-fix artifacts agree here, and that agreement is
// the invariant being pinned.
func TestHXC348Sentinel_LookalikeNamesAreStillRefused(t *testing.T) {
	for _, model := range hxc348SentinelLookalikes {
		t.Run(model, func(t *testing.T) {
			w := hxc348Post(t,
				gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
				"/v1/chat/completions",
				api.ChatCompletionRequest{
					Model:    model,
					Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
				})
			if w.Code != http.StatusNotFound {
				t.Errorf("model %q: status = %d, want 404 — this name merely CONTAINS "+
					"a sentinel, it is not one, and letting it through would re-open "+
					"HXC-348 for every id with a sentinel-shaped prefix; body: %s",
					model, w.Code, w.Body.String())
			}
		})
	}
}

// ── The Anthropic wire ───────────────────────────────────────────────────

// /v1/messages is a FOURTH distinct path into the same Completer, and neither
// the original HXC-348 tests nor the OpenAI-wire cases above touch it. The
// consuming repository's CHANGELOG names it explicitly — the alias contract
// holds "on both the OpenAI- and Anthropic-compatible wires, streaming and
// non-streaming" — so leaving it uncovered would repeat exactly the gap that
// let the `auto` regression ship: a guard on the paths someone happened to
// think of, and silence on the one they did not.
func TestHXC348Sentinel_Messages_SentinelIsHonoured(t *testing.T) {
	for _, sentinel := range hxc348Sentinels {
		t.Run(sentinel, func(t *testing.T) {
			hxc348AssertSentinelHonoured(t, "POST /v1/messages", sentinel,
				hxc348Post(t,
					gateway.HandleMessages(hxc348Chain(t)),
					"/v1/messages",
					map[string]any{
						"model":      sentinel,
						"max_tokens": 16,
						"messages":   []map[string]string{{"role": "user", "content": "Hello"}},
					}))
		})
	}
}

func TestHXC348Sentinel_MessagesStreaming_SentinelIsHonoured(t *testing.T) {
	for _, sentinel := range hxc348Sentinels {
		t.Run(sentinel, func(t *testing.T) {
			hxc348AssertSentinelHonoured(t, "POST /v1/messages (stream)", sentinel,
				hxc348Post(t,
					gateway.HandleMessages(hxc348Chain(t)),
					"/v1/messages",
					map[string]any{
						"model":      sentinel,
						"max_tokens": 16,
						"stream":     true,
						"messages":   []map[string]string{{"role": "user", "content": "Hello"}},
					}))
		})
	}
}

// The Anthropic wire must refuse an unserved name exactly as the OpenAI wire
// does. Without this, the sentinel exemption could be implemented on this path
// as "never refuse anything" and still satisfy the case above.
func TestHXC348Sentinel_Messages_UnservedNamesStayRefused(t *testing.T) {
	for _, model := range []string{"gpt-4o", "claude-opus-4", "auto-gpt", "default-model"} {
		t.Run(model, func(t *testing.T) {
			w := hxc348Post(t,
				gateway.HandleMessages(hxc348Chain(t)),
				"/v1/messages",
				map[string]any{
					"model":      model,
					"max_tokens": 16,
					"messages":   []map[string]string{{"role": "user", "content": "Hello"}},
				})
			if w.Code != http.StatusNotFound {
				t.Errorf("model %q on /v1/messages: status = %d, want 404 — the "+
					"Anthropic wire must refuse an unserved model exactly as the "+
					"OpenAI wire does; body: %s", model, w.Code, w.Body.String())
			}
		})
	}
}

// A space-padded sentinel never reaches the chain: the gateway's model-name
// charset validation rejects it with 400 first. This is pre-existing behaviour
// that predates HXC-348 and is NOT changed here — a model name carrying spaces
// is malformed, and relaxing the validator to accept it would weaken an
// unrelated guard to serve a rendering nothing documents. The case is pinned so
// the boundary is a recorded FACT rather than an assumption, and so a future
// change that quietly relaxes the charset rule is caught here.
//
// It does not consult redMode: both polarities agree, which is what makes it
// a boundary rather than a defect.
func TestHXC348Sentinel_SpacePaddedSentinelIsRejectedAtValidation(t *testing.T) {
	w := hxc348Post(t,
		gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
		"/v1/chat/completions",
		api.ChatCompletionRequest{
			Model:    "  auto  ",
			Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
		})
	if w.Code != http.StatusBadRequest {
		t.Errorf("space-padded sentinel: status = %d, want %d — a model name with "+
			"spaces is malformed and is refused by charset validation before the "+
			"chain sees it; body: %s", w.Code, http.StatusBadRequest, w.Body.String())
	}
	if strings.Contains(w.Body.String(), "model_not_found") {
		t.Errorf("space-padded sentinel: rejected as model_not_found rather than as a "+
			"malformed name, which would misreport a validation defect as a missing "+
			"model: %s", w.Body.String())
	}
}

// The canonical unserved ids from the live probe stay refused. This is the
// original HXC-348 guarantee restated at this file's boundary, so a change
// that widened the sentinel set until everything passed would fail HERE rather
// than quietly satisfying the honoured-sentinel cases above.
func TestHXC348Sentinel_ForeignVendorModelsStayRefused(t *testing.T) {
	for _, model := range []string{"gpt-4o", "claude-opus-4", "definitely-not-a-real-model-zzz"} {
		t.Run(model, func(t *testing.T) {
			w := hxc348Post(t,
				gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
				"/v1/chat/completions",
				api.ChatCompletionRequest{
					Model:    model,
					Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
				})
			if w.Code != http.StatusNotFound {
				t.Errorf("model %q: status = %d, want 404 — the sentinel pass-through "+
					"must not weaken the refusal HXC-348 exists to deliver; body: %s",
					model, w.Code, w.Body.String())
			}
		})
	}
}
