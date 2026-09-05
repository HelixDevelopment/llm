package laneboot_test

// OPEN-26 guard: the measured model-selection decision lives in internal/
// laneboot, shared by all four *-boot lanes. It was extracted from four
// byte-identical copies (abbc0d2); the drift risk is a lane quietly growing
// its own copy of the decision again — the duplication the extraction
// removed. This guard fails the moment any lane re-introduces the decision's
// entry points instead of routing through laneboot.Decide.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// decisionEntryPoints are the symbols that only the shared decision may use.
// A lane that contains any of them has re-grown the copy this package exists
// to deduplicate.
var decisionEntryPoints = []string{
	"selection.Select(",
	"familyRefusalReport(",
	"func declaredUsage(",
}

func TestLanesDoNotReimplementTheDecision(t *testing.T) {
	lanes := []string{"agentgen-boot", "imagegen-boot", "videogen-boot", "visiongen-boot"}
	scanned := 0
	for _, lane := range lanes {
		paths, err := filepath.Glob(filepath.Join("..", "..", "cmd", lane, "*.go"))
		if err != nil {
			t.Fatalf("glob %s sources: %v", lane, err)
		}
		if len(paths) == 0 {
			t.Fatalf("no Go sources found for lane %s — the guard would prove nothing", lane)
		}
		for _, p := range paths {
			if strings.HasSuffix(p, "_test.go") {
				continue
			}
			b, err := os.ReadFile(p)
			if err != nil {
				t.Fatalf("read %s: %v", p, err)
			}
			scanned++
			for _, sym := range decisionEntryPoints {
				if strings.Contains(string(b), sym) {
					t.Fatalf("%s contains %q — lane %s has re-implemented the shared "+
						"selection decision instead of calling laneboot.Decide (OPEN-26)",
						p, sym, lane)
				}
			}
		}
	}
	if scanned == 0 {
		t.Fatal("no lane sources scanned; the guard would prove nothing")
	}
}
