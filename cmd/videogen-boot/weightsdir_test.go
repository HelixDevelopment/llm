package main

// EX-28 guard for the VIDEO-GENERATION boot path: the storage axis of the
// measured decision is resolved on the filesystem that will actually hold the
// lane's weights — the named compose volume mounted at the runtime's HF cache
// — never passed as "" and left for capability.StoragePathForWeights to
// answer from the process working directory.
//
// POLARITY (§11.4.115). One source, two roles, switched by RED_MODE:
//
//	RED_MODE=1 — reproduce the defect on the pre-fix artifact: chooseModel
//	             passed an empty weights dir to laneboot.Decide, so the
//	             storage axis described whatever filesystem the boot command
//	             ran on (measured: 73x different fit answers between /dev/shm
//	             and the volume's filesystem on one host).
//	RED_MODE=0 — the standing guard (default): the empty string is ABSENT
//	             from the Decide call and the dir is resolved through
//	             laneboot.ResolveWeightsDir, whose own seam tests pin the
//	             priority order (env override → volume mountpoint → runtime
//	             storage root → refuse, never the working directory).

import (
	"context"
	"os"
	"strings"
	"testing"
)

// emptyWeightsDirCall is the defect shape: Decide asked to measure storage on
// nothing, which the capability layer silently answered from the cwd.
const emptyWeightsDirCall = `laneboot.Decide(ctx, family, "", pin, forbidKey)`

func TestWeightsDirIsResolvedNotEmpty(t *testing.T) {
	sources := productionSources(t)
	joined := ""
	empty := false
	for path, src := range sources {
		joined += "\n" + src
		if strings.Contains(src, emptyWeightsDirCall) {
			empty = true
			t.Logf("empty weights dir passed to Decide in %s", path)
		}
	}
	resolves := strings.Contains(joined, "laneboot.ResolveWeightsDir") &&
		strings.Contains(joined, "weightsDir(ctx)")

	if redMode() {
		if !empty {
			t.Fatal("RED_MODE=1: expected the empty-weightsDir Decide call on the pre-fix artifact")
		}
		return
	}
	if empty {
		t.Fatal("a lane source passes an empty weights dir to Decide — EX-28: the storage axis " +
			"is measured on the process working directory, not the weights' filesystem")
	}
	if !resolves {
		t.Fatal("no lane source resolves the weights dir through laneboot.ResolveWeightsDir " +
			"and hands it to Decide — the EX-28 resolution is not wired")
	}
}

func TestWeightsDirOperatorOverrideWins(t *testing.T) {
	// The env override is the first priority of the shared resolver and is
	// honoured without consulting any container runtime, so this test is
	// hermetic (§11.4.50).
	override := t.TempDir()
	t.Setenv(weightsDirKey, override)

	dir, err := weightsDir(context.Background())
	if err != nil {
		t.Fatalf("weightsDir with the operator override set: %v", err)
	}
	if dir != override {
		t.Fatalf("operator override not honoured: got %q, want %q", dir, override)
	}
	if wd, werr := os.Getwd(); werr == nil && dir == wd {
		t.Fatal("weightsDir returned the working directory — the EX-28 fallback")
	}
}

func TestWeightsVolumeMatchesComposeFile(t *testing.T) {
	// The volume the resolver asks the runtime about and the volume the
	// compose file actually mounts must be the same name, or the axis is
	// measured on a filesystem the weights never land on.
	b, err := os.ReadFile("compose.videogen.yml")
	if err != nil {
		t.Fatalf("read compose.videogen.yml: %v", err)
	}
	if !strings.Contains(string(b), weightsVolume) {
		t.Fatalf("compose.videogen.yml does not reference %q — resolver and compose file have drifted apart",
			weightsVolume)
	}
	// The mount is the runtime's HF cache; the compose file must say so.
	if !strings.Contains(string(b), "/models/hf") {
		t.Fatal("compose.videogen.yml no longer mounts the weights volume at /models/hf — " +
			"the resolver and the compose mount point have drifted apart")
	}
}
