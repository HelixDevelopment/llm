package brain

import "strings"

// A DELEGATION SENTINEL is a reserved model value meaning "you choose" — the
// caller has expressed no preference and is asking the deployment to pick.
//
// # Why this exists
//
// [ErrModelNotFound] and the refusals that raise it (fallback.Chain's pin
// miss, and Router.Route step 4) derive "this deployment does not serve that"
// from a naming-registry miss. That inference is correct for a model NAME and
// wrong for a KEYWORD: `auto` was never meant to resolve to anything, so a
// registry miss on it carries no information at all. Refusing it turns the
// project's own canonical first API call into a 404:
//
//	website/content/_index.md:29                     {"model":"auto", ...}
//	docs/courses/01-getting-started/lesson-01-introduction.md:131
//	docs/courses/01-getting-started/lesson-03-first-api-call.md:202
//	challenges/banks/performance/nonblocking.yaml:34,65
//	tests/e2e/e2e_test.go:60, tests/e2e/multi_provider_e2e_test.go:24
//	tests/stress/stress_test.go:34, tests/integration/auth_test.go:51
//
// That regression is exactly what [IsDelegationSentinel] prevents, and it is
// the reason the empty string was already exempted one branch away: an empty
// model and `auto` are the SAME request wearing two spellings, and the code
// only ever recognised one of them.
//
// # Why this does not weaken the refusal
//
// HXC-348 exists to stop a caller who names a SPECIFIC model being answered
// by a different one behind a 200. A sentinel makes no such claim — it
// delegates the choice — so answering it from the score-ordered chain is
// obedience, not substitution. `gpt-4o`, `claude-opus-4` and
// `definitely-not-a-real-model-zzz` are unaffected and still refused.
//
// # Why the match is exact, not a prefix
//
// A prefix or substring match would let `auto-gpt`, `automatic` and
// `default-model` through and silently re-open HXC-348 for every id with a
// sentinel-shaped prefix. Those are model NAMES that merely contain a
// keyword, and they are refused like any other unserved name. The gateway
// tests pin them as the negative control.
//
// # Why a registered model of the same name still wins
//
// Both call sites consult this ONLY after the pin has already missed, so a
// deployment that genuinely registers a model called `auto` resolves it
// normally and never reaches here. The sentinel is a fallback reading of the
// value, never an override of a real one.
//
// # Why one definition, shared
//
// fallback.Chain and Router.Route are two independent refusal sites for one
// condition. model_not_found.go records why the ERROR is a single exported
// sentinel rather than two message strings — "two independently-worded copies
// of one condition is exactly how the sentinel would stop meaning one thing" —
// and the exemption is the same argument in the same shape: two copies of the
// keyword list would drift, and the drift would be invisible until one path
// 404'd a request the other answered.
var delegationSentinels = map[string]struct{}{
	// `auto` is the project's documented "let the gateway pick" value, and is
	// the same convention the deployment already uses for every other
	// choose-for-me setting (HELIX_CONTAINER_RUNTIME, HELIX_SCHEDULE_STRATEGY,
	// HELIX_LLM_DEFAULT_PROVIDER all take `auto`). Evidence: 15 occurrences
	// across the website, both getting-started lessons, a challenge bank and
	// four test files.
	"auto": {},

	// `default` is a SHIPPED, RELEASE-NOTED alias, not an inference. The
	// consuming repository's CHANGELOG.md documents it by name under Fixed —
	// "A client requesting an alias (e.g. \"model\":\"default\") now gets back
	// the model that actually served the request, on both the OpenAI- and
	// Anthropic-compatible wires, streaming and non-streaming — not the alias
	// echoed back." That sentence is this exemption's contract: `default` must
	// reach the chain, and the response must name the model that answered.
	// Three QA capture scripts post it against a live gateway
	// (scripts/qa/capture_feat_{helixllm_a21ad7ca,azureguard_eb233785,
	// wirefacade_51c058b1}.sh), and docs/OPERATOR_GUIDE.md:419 uses it too.
	"default": {},
}

// IsDelegationSentinel reports whether a requested model value is a reserved
// "you choose" keyword rather than a claim on a specific model.
//
// The empty string is a sentinel by the same logic — it is the original
// spelling of "no preference" — so callers may use this as the single
// no-preference predicate instead of testing emptiness separately.
//
// Matching is case-folded and space-trimmed because a sentinel is a keyword,
// not an identifier: `Auto` and `auto` express the same delegation, and
// honouring one while refusing the other would be a distinction with nothing
// behind it. Trimming is defence in depth at this layer — the gateway's
// model-name charset validation already refuses a space-padded value with 400
// before either refusal site is reached.
func IsDelegationSentinel(model string) bool {
	trimmed := strings.TrimSpace(model)
	if trimmed == "" {
		return true
	}
	_, ok := delegationSentinels[strings.ToLower(trimmed)]
	return ok
}
