package laneboot

// EX-28: where the storage axis is measured.
//
// Videogen and imagegen keep their weights in a NAMED COMPOSE VOLUME
// (helixllm-videogen-cache / helixllm-imagegen-cache), mounted at the
// runtime's HF cache. Passing an empty WeightsDir to Decide used to make
// capability.StoragePathForWeights fall back to the process working
// directory — so the axis that decides whether the weights fit described
// whatever filesystem the boot command happened to run on (measured: a 73x
// different answer between /dev/shm and / on one host). A wrong number is
// worse than no number (FR-056).
//
// ResolveWeightsDir asks the container runtime where the named volume lives,
// in priority order:
//
//  1. The operator's env override (lane-specific, e.g. VIDEOGEN_WEIGHTS_DIR)
//     names the host directory that will hold the weights.
//  2. The runtime is asked for the volume's mountpoint. Rootless podman is
//     probed first (§11.4.161), docker as fallback.
//  3. Before first acquisition the volume does not exist yet; the runtime's
//     storage root is the honest answer — the named volume WILL be created on
//     that filesystem.
//  4. No runtime can say → refuse, naming the env override as the remedy.
//     The working directory is never the silent fallback.
//
// Every probe is bounded by weightsDirProbeTimeout so a wedged daemon cannot
// stall the decision, and the whole resolution honours the caller's context.

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

// weightsDirProbeTimeout bounds every runtime probe below.
const weightsDirProbeTimeout = 10 * time.Second

// containerRuntimes is the probe order: rootless podman first (§11.4.161),
// docker as fallback. A host with both gets the podman answer.
var containerRuntimes = []string{"podman", "docker"}

// lookPath is the seam tests replace; it reports whether a runtime binary is
// on PATH.
var lookPath = exec.LookPath

// weightsDirExec is the runtime-query seam tests replace. The real
// implementation trims surrounding whitespace from the runtime's answer.
var weightsDirExec = func(ctx context.Context, runtime string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, runtime, args...)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("%s %v: %w", runtime, args, err)
	}
	return strings.TrimSpace(string(out)), nil
}

// ResolveWeightsDir resolves the host filesystem the named volume's weights
// will actually occupy — never the process working directory. See the file
// comment for the priority order; every failure path refuses.
func ResolveWeightsDir(ctx context.Context, envKey, volumeName string) (string, error) {
	if dir := strings.TrimSpace(os.Getenv(envKey)); dir != "" {
		return dir, nil
	}
	probeCtx, cancel := context.WithTimeout(ctx, weightsDirProbeTimeout)
	defer cancel()
	tried := make([]string, 0, len(containerRuntimes))
	for _, rt := range containerRuntimes {
		if _, err := lookPath(rt); err != nil {
			continue
		}
		tried = append(tried, rt)
		// The answer is trimmed here, not only inside the exec seam: a
		// whitespace-only answer means "the runtime had nothing usable",
		// whatever the seam returned.
		if mpOut, err := weightsDirExec(probeCtx, rt,
			"volume", "inspect", volumeName, "--format={{.Mountpoint}}"); err == nil {
			if mp := strings.TrimSpace(mpOut); mp != "" {
				return mp, nil
			}
		}
		if rootOut, err := weightsDirExec(probeCtx, rt, storageRootArgs(rt)...); err == nil {
			if root := strings.TrimSpace(rootOut); root != "" {
				return root, nil
			}
		}
	}
	return "", fmt.Errorf("cannot determine where the %q volume will hold its weights: no container "+
		"runtime answered (tried: %s). Set %s to the host directory that will hold the weights",
		volumeName, strings.Join(tried, ", "), envKey)
}

// storageRootArgs asks the runtime where it keeps its volume data. podman and
// docker expose the root under different info fields.
func storageRootArgs(runtime string) []string {
	if runtime == "docker" {
		return []string{"info", "--format={{.DockerRootDir}}"}
	}
	return []string{"info", "--format={{.Store.GraphRoot}}"}
}
