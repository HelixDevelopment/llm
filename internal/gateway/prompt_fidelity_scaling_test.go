package gateway_test

// Regression guard for the silent prompt-truncation defect on the OpenAI-compat
// route (POST /v1/chat/completions).
//
// THE DEFECT (measured 2026-09-07 against the live gateway on :8443, backend
// llama-server :18434 with --ctx-size 32768):
//
//	content_chars=3000  prompt_tokens=1559
//	content_chars=3900  prompt_tokens=2009
//	content_chars=3990  prompt_tokens=2054
//	content_chars=4000  prompt_tokens=2059   <-- knee
//	content_chars=8000  prompt_tokens=2060
//	content_chars=20000 prompt_tokens=2060   <-- dead flat, ~90% of input dropped
//
// The gateway answered HTTP 200 with a confident reply each time. The knee sits
// at exactly 4000 CHARACTERS, not at a token count, which is what identified the
// cause as a hardcoded char cap rather than a real context-window limit. The
// Anthropic route (/v1/messages) on the SAME process, SAME backend and SAME
// model scaled honestly over the identical sweep (3000->1530, 8000->4030,
// 20000->10030 input_tokens), which isolated the defect to the OpenAI path.
//
// WHAT THIS TEST ASSERTS: the property that was broken — that the prompt
// actually delivered to the provider SCALES with the prompt the caller sent.
// It measures the captured InternalChatRequest rather than reported
// prompt_tokens so that it is hermetic, but the two are equivalent here:
// token count is monotone in character count for a fixed filler, so a plateau
// in delivered characters is exactly the plateau observed in prompt_tokens.
//
// POLARITY SWITCH (§11.4.115). One source, two roles:
//
//	RED_MODE=1  reproduce-and-assert-the-defect-is-PRESENT (pre-fix baseline)
//	RED_MODE=0  (default) standing regression guard: defect is ABSENT
//
// Run RED against the unfixed artifact:  RED_MODE=1 go test -run PromptFidelity
// Run GREEN against the fixed artifact:            go test -run PromptFidelity

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/HelixDevelopment/HelixLLM/internal/gateway"
	"github.com/HelixDevelopment/HelixLLM/pkg/api"
	"github.com/HelixDevelopment/HelixLLM/pkg/types"
)

// This file reuses the package-level redMode() helper declared in
// provider_unavailable_test.go: RED_MODE=1 selects defect-reproduction
// polarity, unset/0 selects the standing GREEN regression guard.

// deliveredPromptChars drives HandleChatCompletions with a single user message
// of n characters and returns how many characters of prompt actually reached
// the provider.
func deliveredPromptChars(t *testing.T, n int) int {
	t.Helper()

	mock := &mockBrainProvider{
		name:      "openai",
		available: true,
		models:    []string{"gpt-4o"},
		response: &types.InternalChatResponse{
			ID:           "chatcmpl-fidelity",
			Model:        "gpt-4o",
			Message:      types.InternalMessage{Role: types.RoleAssistant, Content: "ok"},
			FinishReason: "stop",
			Provider:     types.ProviderOpenAI,
		},
	}
	b := newTestBrain(mock)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/v1/chat/completions", gateway.HandleChatCompletions(b, nil, nil))

	// Same filler shape as the live measurement above.
	content := strings.Repeat("a b c d e f g h ", n/16+2)[:n]

	body, _ := json.Marshal(api.ChatCompletionRequest{
		Model:    "gpt-4o",
		Messages: []api.ChatMessage{{Role: "user", Content: content}},
	})
	w := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("n=%d: status = %d, want 200; body = %s", n, w.Code, w.Body.String())
	}
	if mock.capturedReq == nil {
		t.Fatalf("n=%d: provider was never called", n)
	}

	// Count only user-role content: the gateway legitimately REPLACES the
	// system prompt (a separate design decision), and including it would add
	// a constant that muddies the delivered-vs-sent ratio.
	total := 0
	for _, m := range mock.capturedReq.Messages {
		if m.Role == types.RoleUser {
			total += len(m.Content)
		}
	}
	return total
}

// TestPromptFidelity_ScalesWithInput is the core property: an 5x larger prompt
// must deliver a substantially larger prompt to the provider. Under the defect
// the delivered size is pinned at the 4000-char cap and does not move at all.
func TestPromptFidelity_ScalesWithInput(t *testing.T) {
	const small, large = 4000, 20000

	gotSmall := deliveredPromptChars(t, small)
	gotLarge := deliveredPromptChars(t, large)

	t.Logf("delivered user-prompt chars: sent=%d -> delivered=%d ; sent=%d -> delivered=%d",
		small, gotSmall, large, gotLarge)

	if redMode() {
		// RED: prove the defect is present. The signature is that a 5x larger
		// prompt does NOT arrive 5x larger — it stays pinned near the cap.
		// (Asserting literally zero growth would be brittle: the truncator's
		// head/tail joiner shifts the length by a few characters.)
		if gotLarge >= large {
			t.Fatalf("RED_MODE: expected the truncation defect to be PRESENT, but "+
				"the full %d-char prompt was delivered (%d chars). Not reproducible "+
				"on this artifact.", large, gotLarge)
		}
		t.Logf("RED confirmed: sent %d chars, provider received only %d (%.1f%% "+
			"delivered) — input silently discarded while the gateway answered 200",
			large, gotLarge, 100*float64(gotLarge)/float64(large))
		return
	}

	// GREEN: the whole prompt must reach the provider.
	if gotLarge < large {
		t.Fatalf("SILENT TRUNCATION: sent %d chars, provider received only %d "+
			"(%.1f%%). The gateway must deliver the whole prompt or refuse the "+
			"request explicitly — never silently shorten it and answer anyway.",
			large, gotLarge, 100*float64(gotLarge)/float64(large))
	}
	if gotSmall < small {
		t.Errorf("SILENT TRUNCATION at the smaller size too: sent %d chars, "+
			"provider received %d", small, gotSmall)
	}
}

// TestPromptFidelity_NoSilentLossAcrossSweep walks the sweep that exposed the
// knee. Every step must strictly increase; a flat step is silent loss.
func TestPromptFidelity_NoSilentLossAcrossSweep(t *testing.T) {
	if redMode() {
		t.Skip("sweep is the GREEN-side guard; RED baseline is the scaling test")
	}
	sizes := []int{3000, 4000, 4100, 8000, 20000}
	prev := 0
	for _, n := range sizes {
		got := deliveredPromptChars(t, n)
		t.Logf("sent=%6d chars -> delivered=%6d chars", n, got)
		if got < n {
			t.Errorf("input silently discarded at sent=%d: only %d chars delivered "+
				"(%.1f%%)", n, got, 100*float64(got)/float64(n))
		}
		if got <= prev {
			t.Errorf("delivered prompt did not grow at sent=%d (%d chars, previous "+
				"step delivered %d) — plateau means input is being dropped", n, got, prev)
		}
		prev = got
	}
}

// TestPromptFidelity_OverLimitRefusesLoudly pins the other half of the contract:
// when a prompt genuinely cannot be served it must be REFUSED with an error that
// names the numbers, never answered on a shortened prompt. A confident 200 on a
// prompt the backend could not have seen is the worst of the two failures.
func TestPromptFidelity_OverLimitRefusesLoudly(t *testing.T) {
	if redMode() {
		t.Skip("over-limit refusal does not exist pre-fix; GREEN-side guard only")
	}
	t.Setenv("HELIX_LLM_MAX_PROMPT_TOKENS", "1000")

	mock := &mockBrainProvider{
		name:      "openai",
		available: true,
		models:    []string{"gpt-4o"},
		response: &types.InternalChatResponse{
			ID:      "chatcmpl-should-not-be-reached",
			Model:   "gpt-4o",
			Message: types.InternalMessage{Role: types.RoleAssistant, Content: "ok"},
		},
	}
	b := newTestBrain(mock)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/v1/chat/completions", gateway.HandleChatCompletions(b, nil, nil))

	// Far beyond a 1000-token budget.
	content := strings.Repeat("a b c d e f g h ", 8000)
	body, _ := json.Marshal(api.ChatCompletionRequest{
		Model:    "gpt-4o",
		Messages: []api.ChatMessage{{Role: "user", Content: content}},
	})
	w := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)

	if w.Code == 200 {
		t.Fatalf("over-limit prompt was ANSWERED (200) instead of refused; "+
			"body = %s", w.Body.String())
	}
	if mock.capturedReq != nil {
		t.Errorf("over-limit prompt was still dispatched to the provider")
	}

	msg := w.Body.String()
	// The error must name both numbers so the caller can act on it.
	if !strings.Contains(msg, "1000") {
		t.Errorf("refusal does not name the context limit (1000): %s", msg)
	}
	if !strings.Contains(strings.ToLower(msg), "token") {
		t.Errorf("refusal does not describe the size problem: %s", msg)
	}
	t.Logf("refusal body: %s", msg)
}
