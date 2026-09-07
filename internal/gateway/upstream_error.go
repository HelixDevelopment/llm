package gateway

import (
	"log"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/HelixDevelopment/HelixLLM/internal/fallback"
	"github.com/HelixDevelopment/HelixLLM/internal/shared/i18n"
	"github.com/HelixDevelopment/HelixLLM/pkg/api"
)

// Upstream (Completer) failures reach the client through a single funnel so
// that the two things a caller is entitled to know — that the upstream
// failed, and whether retrying could help — are answered consistently, while
// the one thing the caller is NOT entitled to know is answered nowhere.
//
// # The disclosure this replaces
//
// Every Completer-error site used to interpolate err.Error() straight into
// the response body. When a provider's backend is not listening, that error
// carries the backend's address, so a client of the PUBLIC gateway received
// the PRIVATE topology. Measured against the shipped binary:
//
//	$ curl -sS .../v1/chat/completions -d '{"model":"llama-3.1-70b","messages":[]}'
//	HTTP=503 {"error":{"message":"brain error: all providers exhausted, last
//	error: llamacpp: send request: Post
//	\"http://localhost:50052/v1/chat/completions\":
//	dial tcp 127.0.0.1:50052: connect: connection refused", ...}}
//
// The host, the port, and the upstream path are all disclosed, on an
// unauthenticated endpoint, to anyone who sends a malformed request.
//
// # Redaction, not deletion
//
// The detail is not thrown away — it moves to the server log, where the
// operator who needs it can still read it and the client who must not see it
// cannot. Deleting it would trade a disclosure defect for an operability
// one; UpstreamErrorLogDetail exists so that trade is visible and tested.
//
// # Why two messages
//
// An exhausted chain and a provider fault are different answers (see
// completerErrorStatus), so they get different text as well as different
// status: "retry shortly" for the availability condition, "the provider
// failed" for the fault. Collapsing them would tell a client to retry a
// request that cannot succeed, or not to retry one that would.

// UpstreamErrorLogDetail returns the FULL upstream error text, address and
// all, for server-side logging. It is exported so the guard that proves the
// fix redacts rather than deletes can assert on it; nothing in the response
// path may use it.
func UpstreamErrorLogDetail(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// upstreamErrorMessageKey picks the client-safe message for an upstream
// failure, mirroring the split completerErrorStatus makes on status.
//
// The retired-identifier message is the only one of the three that tells the
// caller what to DO. The other two describe a condition the caller can only
// wait out; this one names a remedy the caller can act on immediately, and has
// to, because the identifier carries a digest and no client can derive its
// replacement — only the listing can hand it over. A status alone would leave
// the holder of a retired identifier knowing that it failed and nothing else.
//
// It names the endpoint but no host, port or upstream path, so it keeps the
// redaction this file exists to enforce.
func upstreamErrorMessageKey(err error) string {
	if fallback.IsRetiredIdentifier(err) {
		return i18n.KeyGatewayIdentifierRetired
	}
	if fallback.IsUnservable(err) {
		return i18n.KeyGatewayUpstreamUnavailable
	}
	return i18n.KeyGatewayUpstreamFailed
}

// upstreamErrorTextForLang is the WebSocket-side companion to
// writeUpstreamError. The WS handler negotiates a language once at upgrade
// time and has no per-frame gin.Context, so it translates by language tag
// rather than by request. Same message, same redaction — the WS path used
// to write err.Error() verbatim with no translation at all, which made it
// both the least localised and the most disclosing of the six error sites.
func upstreamErrorTextForLang(lang string, err error) string {
	return gatewayTranslator.T(lang, upstreamErrorMessageKey(err), nil)
}

// UpstreamErrorClientText is the exported view of the client-safe message,
// so the WebSocket guard can assert the frame carries the TRANSLATED text
// rather than merely "something without an address in it". Without it the
// guard would pass on a frame that had been silenced instead of redacted.
func UpstreamErrorClientText(lang string, err error) string {
	return upstreamErrorTextForLang(lang, err)
}

// WriteUpstreamError is the exported funnel, for brain-backed routes that
// live OUTSIDE this package but on the SAME server — /v1/agents/chat and
// /v1/agents/coordinate run the agent loop over the same brain, wrap its
// error with %w, and used to hand the result to the client verbatim. A
// sibling route leaking the address the gateway just stopped leaking would
// make the fix cosmetic, so they share this funnel rather than growing a
// second, drifting copy of the same policy.
func WriteUpstreamError(c *gin.Context, stage string, err error) {
	writeUpstreamError(c, stage, err)
}

// writeUpstreamError answers a Completer failure: the truthful status, a
// client-safe message, and the full detail in the server log.
//
// stage names the call site ("complete" / "stream") so an operator reading
// the log can tell a streaming failure from a non-streaming one — the
// distinction the old KeyGatewayBrainError / KeyGatewayBrainStreamError
// message pair used to carry in the client's body.
func writeUpstreamError(c *gin.Context, stage string, err error) {
	log.Printf("[HelixLLM] upstream %s failed for %s %s: %s",
		stage, c.Request.Method, c.Request.URL.Path, UpstreamErrorLogDetail(err))

	// A context-size refusal is the one upstream failure the caller can act
	// on, and the only one whose detail is safe to relay: it names token
	// counts, and a token count cannot disclose a host, a port or an upstream
	// path. Collapsing it into the generic "the model provider failed"
	// message would tell a caller whose prompt was simply too large that the
	// backend is broken — and leave them no way to find out it was not.
	//
	// Only the extracted count-bearing clause travels; the surrounding error
	// chain, which may carry an address, stays in the log above.
	if detail := contextOverflowDetail(err); detail != "" {
		c.JSON(http.StatusRequestEntityTooLarge, api.ErrorResponse{
			Error: api.ErrorDetail{
				Message: tr(c, i18n.KeyGatewayUpstreamContextExceeded,
					map[string]string{"detail": detail}),
				Type: "invalid_request_error",
			},
		})
		return
	}

	c.JSON(completerErrorStatus(err), api.ErrorResponse{
		Error: api.ErrorDetail{
			Message: tr(c, upstreamErrorMessageKey(err), nil),
			Type:    "server_error",
		},
	})
}
