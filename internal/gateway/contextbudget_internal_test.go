package gateway

import (
	"errors"
	"os"
	"strings"
	"testing"
)

// Guards for the context-budget policy that replaced the silent 4000-character
// prompt cap. These are package-internal because the units under test are the
// unexported decision points; the end-to-end behaviour they support is guarded
// separately in prompt_fidelity_scaling_test.go.

func TestMaxPromptTokens_DefaultAndOverride(t *testing.T) {
	tests := []struct {
		name string
		env  string
		want int
	}{
		{"unset uses the deployment default", "", defaultMaxPromptTokens},
		{"declared budget is honoured", "8192", 8192},
		// A typo must not silently disable the check and reinstate
		// answer-on-a-truncated-prompt.
		{"non-numeric falls back", "not-a-number", defaultMaxPromptTokens},
		{"zero falls back", "0", defaultMaxPromptTokens},
		{"negative falls back", "-5", defaultMaxPromptTokens},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.env == "" {
				t.Setenv(maxPromptTokensEnv, "")
				os.Unsetenv(maxPromptTokensEnv)
			} else {
				t.Setenv(maxPromptTokensEnv, tt.env)
			}
			if got := maxPromptTokens(); got != tt.want {
				t.Errorf("maxPromptTokens() = %d, want %d", got, tt.want)
			}
		})
	}
}

// TestContextOverflowDetail_ExtractsBothNumbers pins the property the client
// depends on: a context refusal must reach them naming BOTH counts.
//
// The JSON-envelope case is the one that actually ships — llama.cpp returns
// this text inside an error object — and it is the case that regressed on the
// first live check, relaying `...try increasing it","type"` because the tail
// was colon-bounded rather than quote-bounded.
func TestContextOverflowDetail_ExtractsBothNumbers(t *testing.T) {
	jsonEnvelope := errors.New(`llamacpp: unexpected status 400: ` +
		`{"error":{"code":400,"message":"request (40059 tokens) exceeds the ` +
		`available context size (32768 tokens), try increasing it",` +
		`"type":"server_error"}}`)

	got := contextOverflowDetail(jsonEnvelope)
	if got == "" {
		t.Fatal("context refusal was not recognised; the caller would have " +
			"received the generic provider-failed message instead")
	}
	for _, want := range []string{"40059", "32768"} {
		if !strings.Contains(got, want) {
			t.Errorf("detail %q does not name %q; the caller cannot act on "+
				"a refusal that omits either number", got, want)
		}
	}
	if strings.Contains(got, `"type"`) {
		t.Errorf("detail %q leaked JSON envelope keys past the closing quote", got)
	}
	if strings.Contains(got, "llamacpp:") {
		t.Errorf("detail %q leaked the provider/transport prefix", got)
	}
}

// TestContextOverflowDetail_DoesNotRelayUnrelatedErrors is the negative half.
// Without it the classifier could pass by relabelling every failure as an
// oversized prompt — and, worse, relay error chains that carry the backend
// address the upstream-error funnel exists to redact.
func TestContextOverflowDetail_DoesNotRelayUnrelatedErrors(t *testing.T) {
	unrelated := []error{
		nil,
		errors.New("llamacpp: send request: Post \"http://localhost:50052/v1/chat/completions\": dial tcp 127.0.0.1:50052: connect: connection refused"),
		errors.New("brain error: all providers exhausted"),
		errors.New("llamacpp: unexpected status 500: internal server error"),
		errors.New("llamacpp: decode response: unexpected EOF"),
	}
	for _, err := range unrelated {
		if got := contextOverflowDetail(err); got != "" {
			t.Errorf("contextOverflowDetail(%v) = %q, want empty: an unrelated "+
				"failure must not be relabelled as an oversized prompt, and its "+
				"error text must not reach the client", err, got)
		}
	}
}

// TestEstimateTokens_ScalesAndNeverZeroForNonEmpty pins the two properties the
// budget check relies on.
func TestEstimateTokens_ScalesAndNeverZeroForNonEmpty(t *testing.T) {
	if got := estimateTokens(""); got != 0 {
		t.Errorf("estimateTokens(\"\") = %d, want 0", got)
	}
	if got := estimateTokens("a"); got < 1 {
		t.Errorf("estimateTokens(%q) = %d, want >= 1: a non-empty prompt that "+
			"estimates to zero would bypass the budget check", "a", got)
	}
	small := estimateTokens(strings.Repeat("x", 1000))
	large := estimateTokens(strings.Repeat("x", 10000))
	if large <= small {
		t.Errorf("estimate did not grow with input: 1000 chars -> %d, "+
			"10000 chars -> %d", small, large)
	}
}
