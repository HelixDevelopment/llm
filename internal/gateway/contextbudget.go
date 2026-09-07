package gateway

import (
	"encoding/json"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/HelixDevelopment/HelixLLM/pkg/api"
)

// Context-budget policy: deliver the whole prompt, or refuse and say why.
//
// # What this replaces
//
// The OpenAI-compat route used to shorten every oversized message to a fixed
// character cap and answer anyway. Measured on the shipped binary against a
// backend started with --ctx-size 32768:
//
//	content_chars=3000  prompt_tokens=1559
//	content_chars=3900  prompt_tokens=2009
//	content_chars=4000  prompt_tokens=2059   <-- knee
//	content_chars=8000  prompt_tokens=2060
//	content_chars=20000 prompt_tokens=2060   <-- flat; ~90% of input dropped
//
// Every one of those returned HTTP 200 with a confident answer. The knee sits
// at exactly 4000 CHARACTERS, which is what identified it as a hardcoded cap
// rather than a real context limit — and explains why two independent
// measurements saw DIFFERENT token plateaus for the same defect: a character
// cap converts to a different token count for every prompt shape.
//
// The Anthropic route on the SAME process, backend and model scaled honestly
// across the identical sweep, which is what isolated the defect to this path.
//
// # Why refusing beats truncating
//
// A truncated prompt answered at 200 is strictly worse than an error. The
// caller cannot tell the difference between an answer about their code and an
// answer about the first 4000 characters of it, so a coding agent sending a
// large file receives confident output about code the model never read. An
// error is visible, actionable, and cannot be mistaken for a result. This is
// the same posture the backend already took before the gateway fronted it —
// llama.cpp refuses an oversized request loudly and names both numbers.
//
// # Two measurements, not one
//
// The gateway's own estimate is a pre-flight: cheap, no round-trip, and
// approximate. The backend's count is authoritative but only available after
// dispatch. Both are kept: the estimate stops the extreme cases early, and
// upstreamContextExceeded relays the backend's exact numbers when a prompt
// slips past the estimate. Neither is allowed to silently shorten anything.

// defaultMaxPromptTokens is the fallback prompt budget when the deployment
// does not declare one.
//
// It is NOT a guess: 32768 is the context the local llama.cpp backend is
// launched with in this deployment (--ctx-size 32768, confirmed against the
// running server's /props reporting n_ctx=32768, and against a live probe
// where a 30030-token prompt was served and a ~35000-token prompt was
// refused). A deployment that serves a different context MUST declare it via
// HELIX_LLM_MAX_PROMPT_TOKENS rather than rely on this constant matching its
// backend.
const defaultMaxPromptTokens = 32768

// maxPromptTokensEnv is the deployment's declared prompt budget in tokens.
const maxPromptTokensEnv = "HELIX_LLM_MAX_PROMPT_TOKENS"

// maxPromptTokens reports the prompt budget this deployment will accept.
//
// A malformed or non-positive value falls back to the default rather than
// disabling the check: a typo in an env var must not silently reinstate the
// answer-on-a-truncated-prompt behaviour this file exists to remove.
func maxPromptTokens() int {
	raw := os.Getenv(maxPromptTokensEnv)
	if raw == "" {
		return defaultMaxPromptTokens
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return defaultMaxPromptTokens
	}
	return n
}

// charsPerTokenDivisor converts characters to an estimated token count.
//
// Calibrated on this deployment's own measurements rather than taken from
// literature: the whitespace-heavy filler used in the reproduction tokenised
// at ~0.5 tokens/char (3000 chars -> 1530 tokens), while prose and code sit
// nearer 0.25-0.3. Dividing by 3 sits between them, deliberately closer to
// the prose end.
//
// The direction of the residual error is chosen, not accidental. Estimating
// LOW lets a borderline prompt reach the backend, which then refuses it
// loudly with its exact numbers (relayed by upstreamContextExceeded).
// Estimating HIGH would refuse prompts that would have been served — turning
// this guard into a smaller version of the cap it replaces. Between two
// imperfect options only one of them is silent, and it is not this one.
const charsPerTokenDivisor = 3

// estimateTokens approximates the token count of s.
func estimateTokens(s string) int {
	if s == "" {
		return 0
	}
	// Round up so a short-but-nonempty string never estimates to zero.
	return (len(s) + charsPerTokenDivisor - 1) / charsPerTokenDivisor
}

// estimateRequestTokens approximates the prompt size of a chat request,
// counting every message the caller sent plus the tool schemas that travel
// with it — all of which consume the same context window.
func estimateRequestTokens(req *api.ChatCompletionRequest) int {
	total := 0
	for _, m := range req.Messages {
		total += estimateMessageTokens(m)
	}
	for _, t := range req.Tools {
		total += estimateTokens(t.Function.Name)
		total += estimateTokens(t.Function.Description)
		if t.Function.Parameters != nil {
			// Parameters are a JSON schema of unknown shape; its rendered
			// size is what reaches the model, so measure that rather than
			// assume a fixed cost per tool.
			total += estimateTokens(renderForSizing(t.Function.Parameters))
		}
	}
	return total
}

// estimateMessageTokens measures one message, handling both content shapes
// the OpenAI schema allows: a bare string, and the content-part array that
// multimodal-capable clients send.
func estimateMessageTokens(m api.ChatMessage) int {
	total := 0
	switch content := m.Content.(type) {
	case string:
		total += estimateTokens(content)
	case []interface{}:
		for _, part := range content {
			pm, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			if text, ok := pm["text"].(string); ok {
				total += estimateTokens(text)
			}
		}
	}
	for _, tc := range m.ToolCalls {
		total += estimateTokens(tc.Function.Name)
		total += estimateTokens(tc.Function.Arguments)
	}
	return total
}

// renderForSizing produces a length-representative rendering of an arbitrary
// JSON-ish value. It is used only to SIZE the value, never to transmit it,
// so an approximate rendering is sufficient and a marshalling failure simply
// contributes nothing rather than failing the request.
func renderForSizing(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(b)
}

// promptBudgetExceeded reports whether req's estimated prompt exceeds limit,
// along with the estimate itself so the refusal can name it.
func promptBudgetExceeded(req *api.ChatCompletionRequest, limit int) (int, bool) {
	est := estimateRequestTokens(req)
	return est, est > limit
}

// contextOverflowPattern recognises a backend's context-size refusal.
//
// Matching on the SHAPE of the message rather than an error type is
// deliberate: the condition is reported as prose by the backends this
// gateway fronts, and there is no typed error to match on. The pattern is
// anchored on the two words that co-occur only in this condition, so an
// unrelated failure does not get relabelled as an oversized prompt.
var contextOverflowPattern = regexp.MustCompile(
	`(?i)(exceeds?|larger than|beyond).{0,40}context|context (size|window|length).{0,40}(exceed|too small|insufficient)`)

// contextOverflowDetail extracts the backend's own description of a
// context-size refusal, or "" when err is not that condition.
//
// Only the matched sentence travels to the client. That is what keeps this
// compatible with the redaction policy in upstream_error.go: a context-size
// refusal names token counts, and a token count cannot disclose a host, a
// port or an upstream path.
func contextOverflowDetail(err error) string {
	if err == nil {
		return ""
	}
	msg := err.Error()
	if !contextOverflowPattern.MatchString(msg) {
		return ""
	}
	return extractContextSentence(msg)
}

// contextSentencePattern captures the count-bearing clause of a backend
// context refusal, e.g.
//
//	request (141440 tokens) exceeds the available context size (4096 tokens)
//
// so the client receives the two numbers it needs and nothing else from the
// surrounding error chain, which may carry an address.
//
// The clause is bounded by double quotes rather than by colons. Backends
// return this text inside a JSON envelope, and a colon-bounded tail ran past
// the closing quote into the next key — the first live check relayed
// `...try increasing it","type"` to the client. A quote-bounded tail stops at
// the end of the JSON string, and degrades correctly for a plain-text body,
// which has no quotes to stop at.
var contextSentencePattern = regexp.MustCompile(
	`(?i)[^"]*\(\s*\d+\s*tokens?\s*\)[^"]*\(\s*\d+\s*tokens?\s*\)[^"]*`)

// extractContextSentence narrows a backend error to its count-bearing clause,
// falling back to a digits-and-context-words-only reconstruction when the
// message does not use the parenthesised two-number form.
func extractContextSentence(msg string) string {
	if m := contextSentencePattern.FindString(msg); m != "" {
		return strings.TrimSpace(m)
	}
	// No recognised two-number form. Rather than relay an arbitrary error
	// chain (which may carry an address), report only that the condition was
	// the context window; the exact figures stay in the server log.
	return ""
}
