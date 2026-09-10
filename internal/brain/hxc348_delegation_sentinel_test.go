// HXC-348 follow-up — the SECOND refusal site must honour the same sentinels.
//
// # Why this file exists separately from the gateway's
//
// One condition — "this deployment does not serve that model" — is raised
// from TWO independent places: fallback.Chain's pin miss (the production
// serving path cmd/helixllm wires) and Router.Route step 4 (reached through
// Brain.Complete / Brain.CompleteStream). The gateway tests drive the first.
// Nothing drove the second, so a sentinel exemption applied to only one of
// them would leave `auto` answered on one path and 404'd on the other, with
// the divergence invisible until a caller hit the unguarded one.
//
// delegation_sentinel.go states why the KEYWORD LIST is shared rather than
// copied. This file is the executable half of that argument: it pins the
// router's agreement with the gateway, so a future edit that re-introduces a
// second copy of the list and lets it drift fails here.
//
// # Polarity switch (§11.4.115)
//
// RED_MODE=1 asserts the regression IS present at this site — Route refuses a
// sentinel — so a run against the unfixed tree PASSES and proves the
// reproduction real. RED_MODE=0 (default) is the standing GREEN guard.
package brain_test

import (
	"os"
	"testing"

	"github.com/HelixDevelopment/HelixLLM/internal/brain"
	"github.com/HelixDevelopment/HelixLLM/pkg/types"
)

// sentinelRedMode is this file's own polarity switch. It is named distinctly
// because package brain_test already carries a redMode(*testing.T) with a
// different signature (loopback_serving_host_test.go), and shadowing it would
// silently change which switch an unrelated test reads.
func sentinelRedMode() bool { return os.Getenv("RED_MODE") == "1" }

// hxc348SentinelValues is the documented delegation set plus its case
// renderings — the same set the gateway file drives, asserted here at the
// other refusal site. The empty string is included because it is the original
// spelling of "no preference" and IsDelegationSentinel is meant to be the
// single predicate for all of them.
var hxc348SentinelValues = []string{"", "auto", "AUTO", "Auto", "default", "DEFAULT"}

// The unit-level contract of the shared predicate. It is asserted directly,
// not only through Route, because both refusal sites depend on it and a
// defect here would surface as two unrelated-looking routing bugs.
func TestHXC348Sentinel_IsDelegationSentinel(t *testing.T) {
	for _, v := range hxc348SentinelValues {
		if !brain.IsDelegationSentinel(v) {
			t.Errorf("IsDelegationSentinel(%q) = false, want true — this value "+
				"delegates the choice and names no model to be missing", v)
		}
	}
	// The negative control. A name that merely CONTAINS a sentinel is a model
	// name, and matching it would re-open HXC-348 for every id with a
	// sentinel-shaped prefix.
	for _, v := range []string{
		"auto-gpt", "automatic", "autopilot-7b",
		"default-model", "gpt-auto", "auto/llama",
		"gpt-4o", "claude-opus-4", "definitely-not-a-real-model-zzz",
	} {
		if brain.IsDelegationSentinel(v) {
			t.Errorf("IsDelegationSentinel(%q) = true, want false — this is a model "+
				"NAME, and treating it as a keyword would silently substitute a "+
				"different model for a caller who named a specific one", v)
		}
	}
}

// Route must answer a sentinel exactly as it answers an empty model.
func TestHXC348Sentinel_Route_SentinelStillFallsBack(t *testing.T) {
	for _, sentinel := range hxc348SentinelValues {
		t.Run("model="+sentinel, func(t *testing.T) {
			r := brain.NewRouter("llamacpp")
			r.Register("llamacpp", newMock("llamacpp", true, "qwen2.5-coder-3b"))

			p, err := r.Route(&types.InternalChatRequest{Model: sentinel})

			if sentinelRedMode() {
				if sentinel == "" {
					t.Skip("the empty model was already exempt before the regression")
				}
				if err == nil || !brain.IsModelNotFound(err) {
					t.Errorf("RED_MODE expects Route(%q) to be refused as model_not_found, "+
						"got provider=%v err=%v", sentinel, p, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("Route(%q) returned error %v; a delegation sentinel names no "+
					"model, so it must be routed exactly as an empty model is",
					sentinel, err)
			}
			if p == nil {
				t.Fatalf("Route(%q) returned a nil provider and no error", sentinel)
			}
			if p.Name() != "llamacpp" {
				t.Errorf("Route(%q) provider = %q, want %q", sentinel, p.Name(), "llamacpp")
			}
		})
	}
}

// The refusal HXC-348 delivers is unchanged at this site. Both polarities
// agree here, so this case does not consult the switch — a change that widened
// the sentinel set until everything routed would fail HERE rather than quietly
// satisfying the case above.
func TestHXC348Sentinel_Route_UnservedNamesStayRefused(t *testing.T) {
	for _, model := range []string{
		"gpt-4o", "claude-opus-4", "definitely-not-a-real-model-zzz",
		"auto-gpt", "automatic", "default-model",
	} {
		t.Run(model, func(t *testing.T) {
			r := brain.NewRouter("llamacpp")
			r.Register("llamacpp", newMock("llamacpp", true, "qwen2.5-coder-3b"))

			p, err := r.Route(&types.InternalChatRequest{Model: model})
			if err == nil {
				t.Fatalf("Route(%q) returned provider %q; a caller that named a specific "+
					"unserved model must be refused, never answered by another model",
					model, p.Name())
			}
			if !brain.IsModelNotFound(err) {
				t.Errorf("Route(%q): IsModelNotFound(%v) = false, want true", model, err)
			}
		})
	}
}
