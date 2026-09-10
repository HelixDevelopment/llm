package gateway

import (
	"net/http"

	"github.com/HelixDevelopment/HelixLLM/internal/brain"
	"github.com/HelixDevelopment/HelixLLM/internal/fallback"
)

// completerErrorStatus maps a Completer failure onto the HTTP status that
// tells the truth about it.
//
// An unservable request means the service cannot answer it RIGHT NOW — either
// every configured provider was skipped, unavailable, or failed (an exhausted
// chain), or the SPECIFIC model the request pinned has no serving provider up.
// Both are availability conditions, which is 503 Service Unavailable
// (RFC 9110 §15.6.4). Clients, load
// balancers, and readiness probes all read 503 as "retry with backoff" and
// 500 as "this build is broken"; reporting a warming-up or unreachable
// backend as 500 tells every one of them the wrong thing.
//
// A model name NO registered provider serves is checked first, and is not an
// availability condition either: no amount of backoff makes `gpt-4o` appear on
// a llama.cpp-only deployment. It gets 404 for the same reason the retired
// identifier below does — it is a name that identifies nothing here, and 503
// would tell a correct client to retry it forever. Before this check the
// condition did not reach here at all: the router and the fallback chain both
// substituted a different model and answered 200 (HXC-348).
//
// A RETIRED identifier is checked next, and is the other case that is not an
// availability condition: the request named an identifier this
// deployment published before its serving host was renamed and will never
// publish again. 503 tells a client to retry with backoff, and a correct client
// obeying that against a name that can never resolve retries forever. It gets
// 404 — the status this gateway already uses for every other "the name you gave
// identifies nothing here" answer (an unknown consumer, an unknown host, a
// model id absent from the listing), so a client's handling of the condition
// does not depend on which endpoint reported it. 410 would also be defensible
// on its own terms, but it would be a SECOND spelling of an answer this server
// already has one for.
//
// Anything else — a provider that WAS reached and returned a fault — stays
// 500, because that genuinely is an internal failure rather than an
// availability condition. Collapsing the two is what this function exists to
// prevent.
func completerErrorStatus(err error) int {
	if brain.IsModelNotFound(err) {
		return http.StatusNotFound
	}
	if fallback.IsRetiredIdentifier(err) {
		return http.StatusNotFound
	}
	if fallback.IsUnservable(err) {
		return http.StatusServiceUnavailable
	}
	return http.StatusInternalServerError
}
