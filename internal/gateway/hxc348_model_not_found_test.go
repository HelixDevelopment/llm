// HXC-348 — an unresolvable model must be REFUSED, never substituted.
//
// # The defect this reproduces
//
// The gateway failed OPEN on a model id nothing in the deployment serves.
// Measured against the running build:
//
//	model="definitely-not-a-real-model-zzz" -> HTTP 200, answered by
//	                                          qwen2.5-coder-3b-instruct-q4_k_m
//	model="gpt-4o" / "claude-opus-4"        -> HTTP 200, same backend
//	model="../../etc/passwd"                -> HTTP 400 (validation DOES run)
//
// The 200s are real answers, not an artefact of a broken harness: a bad token
// still returned 401, malformed JSON 400, and a body missing `messages` 400.
//
// The caller names a model, receives a confident completion from a DIFFERENT
// model, and has no way to detect the substitution — the silent misroute the
// naming layer's own comments call out (internal/brain/naming.go, "fall
// through to whichever provider the router reached last") while leaving the
// unknown-name case unguarded.
//
// # Why the fallback itself is not the defect
//
// A caller that names NO model has expressed no preference, and answering it
// from the score-ordered chain is the feature. The defect is answering a
// caller who named something specific with something else. So the fix refuses
// a NON-EMPTY, UNRESOLVED model and leaves the empty-model path alone — which
// is why the two regression guards below are as load-bearing as the 404s.
//
// # Why 404 / model_not_found
//
// "Requested resource does not exist" is 404 in OpenAI's error-codes guide,
// and this gateway already answers 404 for an unknown model id on
// /v1/models/:id (HandleGetModel). 503 would be a lie with a deadline that
// never arrives: it tells a correct client to retry with backoff a name that
// can never resolve — the exact reasoning completer_status.go already applies
// to a retired identifier. The body reuses the shape requestvalidate.go
// established for model-shaped faults (`invalid_request_error`, param
// "model", an OpenAI `code`) and the message key HandleGetModel already uses,
// so nothing new is invented on either axis.
//
// # Polarity switch (§11.4.115)
//
// RED_MODE=1 reproduces the DEFECT on the pre-fix artifact: it asserts the
// broken behaviour (200, answered by a substituted model) is present, so a
// run against the unfixed build PASSES and proves the reproduction is real
// rather than synthetic. RED_MODE=0 (the default) is the standing GREEN
// regression guard: the substitution is ABSENT and the answer is 404.
//
//	RED_MODE=1 go test -run TestHXC348 ./internal/gateway/   # pre-fix
//	           go test -run TestHXC348 ./internal/gateway/   # post-fix
//
// # What this pins, and what it deliberately does not
//
// The stack under test is the PRODUCTION one: a fallback.Chain with the Brain
// installed as its ModelPinner, which is what cmd/helixllm wires (see
// newFallbackChain). A test against a bare *brain.Brain would exercise a
// Completer the shipped binary does not use, and would have reported the
// defect fixed while it was live.
//
// A model that IS known to some registered provider whose backend happens to
// be down stays an availability condition (503) and is NOT touched here.
package gateway_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/HelixDevelopment/HelixLLM/internal/brain"
	"github.com/HelixDevelopment/HelixLLM/internal/fallback"
	"github.com/HelixDevelopment/HelixLLM/internal/gateway"
	"github.com/HelixDevelopment/HelixLLM/pkg/api"
	"github.com/HelixDevelopment/HelixLLM/pkg/types"
)

// hxc348ServedModel is the one model the deployment under test actually
// serves. Every 404 case below names something that is NOT this.
const hxc348ServedModel = "qwen2.5-coder-3b-instruct-q4_k_m"

// hxc348UnknownModel is the exact id the live probe used.
const hxc348UnknownModel = "definitely-not-a-real-model-zzz"

// hxc348Provider is a stand-in backend that serves exactly one model and is
// always up. It answers ANY request handed to it — which is precisely what
// makes it a useful oracle here: if the gateway routes an unknown model to
// it, the substitution shows up as a 200 whose body names the served model
// rather than the requested one.
type hxc348Provider struct{}

func (hxc348Provider) Name() string     { return "fake" }
func (hxc348Provider) Available() bool  { return true }
func (hxc348Provider) Models() []string { return []string{hxc348ServedModel} }

func (hxc348Provider) Complete(_ context.Context, req *types.InternalChatRequest) (*types.InternalChatResponse, error) {
	return &types.InternalChatResponse{
		ID:    "resp-1",
		Model: req.Model,
		Message: types.InternalMessage{
			Role: types.RoleAssistant, Content: "ok",
		},
		FinishReason: "stop",
	}, nil
}

func (hxc348Provider) CompleteStream(_ context.Context, req *types.InternalChatRequest) (<-chan types.StreamChunk, error) {
	ch := make(chan types.StreamChunk, 1)
	ch <- types.StreamChunk{Content: "ok", FinishReason: "stop"}
	close(ch)
	return ch, nil
}

// hxc348DownProvider serves the same model but reports itself unreachable —
// the "known name, backend down" case the fix must leave as an availability
// condition.
type hxc348DownProvider struct{ hxc348Provider }

func (hxc348DownProvider) Available() bool { return false }

// hxc348Chain builds the SHIPPED serving stack: a Chain whose only entry is
// the fake provider, with the Brain installed as ModelPinner. This mirrors
// cmd/helixllm's newFallbackChain — provider map from the Brain, entries set,
// names registered, pinner installed — because the pinner is the component
// whose "nothing serves this" answer the chain used to read as "the caller
// named nothing".
func hxc348Chain(t *testing.T) *fallback.Chain {
	t.Helper()

	b := brain.New(brain.Config{DefaultProvider: "fake"})
	b.RegisterProvider("fake", hxc348Provider{})

	c := fallback.NewChain(b.Providers(), fallback.NewRateLimitTracker(100, 100))
	c.SetEntries([]fallback.ChainEntry{{
		ProviderName:    "fake",
		ModelID:         hxc348ServedModel,
		Score:           1,
		Status:          fallback.EntryActive,
		IsLocalFallback: true,
	}})
	b.RegisterNames()
	c.SetModelPinner(b)
	return c
}

// hxc348Post drives one request through a real gin engine and returns the
// recorder, so both the status AND the body are available to assert on.
func hxc348Post(t *testing.T, h gin.HandlerFunc, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST(path, h)

	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	return w
}

// hxc348AssertRefused applies the polarity switch for an unresolvable model.
//
// GREEN: 404 with an OpenAI-shaped body whose code is model_not_found and
// whose message names the model the caller asked for.
// RED: the pre-fix artifact answers 200 and the body does NOT carry the
// refusal — which is the substitution, observed rather than assumed.
func hxc348AssertRefused(t *testing.T, route string, w *httptest.ResponseRecorder) {
	t.Helper()

	if redMode() {
		if w.Code != http.StatusOK {
			t.Errorf("%s: RED_MODE expects the pre-fix fail-open, got status %d body %s",
				route, w.Code, w.Body.String())
		}
		if strings.Contains(w.Body.String(), "model_not_found") {
			t.Errorf("%s: RED_MODE expects no refusal on the pre-fix artifact, got %s",
				route, w.Body.String())
		}
		return
	}

	if w.Code != http.StatusNotFound {
		t.Errorf("%s: status = %d, want %d — an unresolvable model must be refused, "+
			"never answered by a substituted one; body: %s",
			route, w.Code, http.StatusNotFound, w.Body.String())
		return
	}

	var got api.ErrorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("%s: response body is not an OpenAI error object: %v (%s)",
			route, err, w.Body.String())
	}
	if got.Error.Type != "invalid_request_error" {
		t.Errorf("%s: error.type = %q, want %q", route, got.Error.Type, "invalid_request_error")
	}
	if got.Error.Code == nil || *got.Error.Code != "model_not_found" {
		t.Errorf("%s: error.code = %v, want %q — the code is what an OpenAI client "+
			"switches on", route, got.Error.Code, "model_not_found")
	}
	if got.Error.Param == nil || *got.Error.Param != "model" {
		t.Errorf("%s: error.param = %v, want %q — a client cannot fix a field the "+
			"error does not name", route, got.Error.Param, "model")
	}
	if !strings.Contains(got.Error.Message, hxc348UnknownModel) {
		t.Errorf("%s: error.message = %q, want it to name the requested model %q",
			route, got.Error.Message, hxc348UnknownModel)
	}
}

func TestHXC348_ChatCompletions_UnknownModelIsRefused(t *testing.T) {
	hxc348AssertRefused(t, "POST /v1/chat/completions", hxc348Post(t,
		gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
		"/v1/chat/completions",
		api.ChatCompletionRequest{
			Model:    hxc348UnknownModel,
			Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
		}))
}

// The streaming arm shares Chain.CompleteStream's own pin, so it is a
// SEPARATE call site and needs its own guard: streaming is how a chat client
// actually talks, and a fix that only covered the buffered path would leave
// the defect live for the common case.
func TestHXC348_ChatCompletionsStreaming_UnknownModelIsRefused(t *testing.T) {
	hxc348AssertRefused(t, "POST /v1/chat/completions (stream)", hxc348Post(t,
		gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
		"/v1/chat/completions",
		api.ChatCompletionRequest{
			Model:    hxc348UnknownModel,
			Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
			Stream:   true,
		}))
}

func TestHXC348_Completions_UnknownModelIsRefused(t *testing.T) {
	hxc348AssertRefused(t, "POST /v1/completions", hxc348Post(t,
		gateway.HandleCompletions(hxc348Chain(t)),
		"/v1/completions",
		map[string]any{
			"model":  hxc348UnknownModel,
			"prompt": "Hello",
		}))
}

// A vendor id this deployment does not serve is the same defect wearing a
// plausible name, and it is the one a real client actually sends.
func TestHXC348_ChatCompletions_ForeignVendorModelIsRefused(t *testing.T) {
	for _, model := range []string{"gpt-4o", "claude-opus-4", "llama3.3:70b"} {
		t.Run(model, func(t *testing.T) {
			w := hxc348Post(t,
				gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
				"/v1/chat/completions",
				api.ChatCompletionRequest{
					Model:    model,
					Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
				})
			if redMode() {
				if w.Code != http.StatusOK {
					t.Errorf("RED_MODE expects the pre-fix fail-open for %q, got %d",
						model, w.Code)
				}
				return
			}
			if w.Code != http.StatusNotFound {
				t.Errorf("model %q: status = %d, want 404; body: %s",
					model, w.Code, w.Body.String())
			}
		})
	}
}

// ── Regression guards: the legitimate paths the fix must not break ───────

// An empty model is the caller saying "any model you like". Refusing it would
// be a worse defect than the one being fixed: it breaks every client that
// relies on the documented default, and requestvalidate.go already records
// that an empty model is deliberately NOT a fault.
func TestHXC348_ChatCompletions_EmptyModelStillFallsBack(t *testing.T) {
	w := hxc348Post(t,
		gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
		"/v1/chat/completions",
		api.ChatCompletionRequest{
			Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
		})
	if w.Code != http.StatusOK {
		t.Errorf("empty model: status = %d, want 200 — no preference means the "+
			"score-ordered chain answers; body: %s", w.Code, w.Body.String())
	}
}

// /v1/completions has its own defaulting, and it used to invent a CONCRETE
// model id ("llama-3.1-70b") for a caller that named none. Once an
// unresolvable id is refused, inventing one turns a no-preference request
// into a 404 — so the handler must pass the caller's empty model through and
// let the chain default, exactly as /v1/chat/completions does.
func TestHXC348_Completions_EmptyModelStillFallsBack(t *testing.T) {
	w := hxc348Post(t,
		gateway.HandleCompletions(hxc348Chain(t)),
		"/v1/completions",
		map[string]any{"prompt": "Hello"})
	if w.Code != http.StatusOK {
		t.Errorf("empty model: status = %d, want 200 — a caller that named no model "+
			"must not be answered with model_not_found; body: %s", w.Code, w.Body.String())
	}
}

// The served model must still route, and must still be answered BY that
// model. Without this, a fix that refused everything would pass every case
// above.
func TestHXC348_ChatCompletions_ServedModelStillRoutes(t *testing.T) {
	w := hxc348Post(t,
		gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
		"/v1/chat/completions",
		api.ChatCompletionRequest{
			Model:    hxc348ServedModel,
			Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
		})
	if w.Code != http.StatusOK {
		t.Fatalf("served model: status = %d, want 200; body: %s", w.Code, w.Body.String())
	}
	var resp api.ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("served model: body is not a completion response: %v (%s)",
			err, w.Body.String())
	}
	if resp.Model != hxc348ServedModel {
		t.Errorf("served model: answered by %q, want %q", resp.Model, hxc348ServedModel)
	}
}

// The boundary the fix must NOT cross: a model this deployment DOES serve,
// whose backend happens to be down, is an availability condition. It keeps the
// retryable answer (503) — turning it into 404 would tell a client to stop
// asking for a model that will come back, and would change an answer this fix
// has no business touching.
func TestHXC348_ChatCompletions_ServedModelOnDownBackendStays503(t *testing.T) {
	b := brain.New(brain.Config{DefaultProvider: "fake"})
	b.RegisterProvider("fake", hxc348DownProvider{})

	c := fallback.NewChain(b.Providers(), fallback.NewRateLimitTracker(100, 100))
	c.SetEntries([]fallback.ChainEntry{{
		ProviderName:    "fake",
		ModelID:         hxc348ServedModel,
		Score:           1,
		Status:          fallback.EntryActive,
		IsLocalFallback: true,
	}})
	b.RegisterNames()
	c.SetModelPinner(b)

	w := hxc348Post(t,
		gateway.HandleChatCompletions(c, nil, nil),
		"/v1/chat/completions",
		api.ChatCompletionRequest{
			Model:    hxc348ServedModel,
			Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
		})
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("served model on a down backend: status = %d, want %d — a backend "+
			"that is down is retryable, not a name that does not exist; body: %s",
			w.Code, http.StatusServiceUnavailable, w.Body.String())
	}
	if strings.Contains(w.Body.String(), "model_not_found") {
		t.Errorf("served model on a down backend was reported as model_not_found: %s",
			w.Body.String())
	}
}

func TestHXC348_ChatCompletionsStreaming_ServedModelStillRoutes(t *testing.T) {
	w := hxc348Post(t,
		gateway.HandleChatCompletions(hxc348Chain(t), nil, nil),
		"/v1/chat/completions",
		api.ChatCompletionRequest{
			Model:    hxc348ServedModel,
			Messages: []api.ChatMessage{{Role: "user", Content: "Hello"}},
			Stream:   true,
		})
	if w.Code != http.StatusOK {
		t.Fatalf("served model (stream): status = %d, want 200; body: %s",
			w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "chat.completion.chunk") {
		t.Errorf("served model (stream): body is not an SSE chunk stream: %s", w.Body.String())
	}
}
