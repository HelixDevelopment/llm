package laneboot

// EX-28 guard: the storage axis must be measured on the filesystem that will
// actually hold the lane's weights — the named container volume the compose
// file mounts at the runtime's HF cache — never on the process working
// directory.
//
// THE DEFECT (measured, findings ledger 2026-09-05): videogen-boot and
// imagegen-boot passed an empty WeightsDir to laneboot.Decide, so
// capability.StoragePathForWeights fell back to os.Getwd(). The same host,
// same volume: 15325 MiB measured from /dev/shm vs 1120821 MiB from / — a 73x
// different answer deciding whether the weights fit. The named axis was right
// and the number described the wrong filesystem.
//
// RESOLUTION RULE under test, in priority order:
//
//  1. The operator's env override (lane-specific, e.g. VIDEOGEN_WEIGHTS_DIR)
//     names the host directory that will hold the weights.
//  2. The container runtime is ASKED where the named volume is mounted
//     (`<runtime> volume inspect <name> --format={{.Mountpoint}}`).
//  3. Before first acquisition the volume does not exist yet; the runtime is
//     asked for its storage root (podman: {{.Store.GraphRoot}}, docker:
//     {{.DockerRootDir}}) — the filesystem the volume WILL be created on.
//  4. No runtime can say → REFUSE, naming the env override as the remedy.
//     The working directory is never the silent fallback: a wrong number is
//     worse than no number (FR-056).
//
// The exec seam and the PATH lookup are replaced in these tests, so they are
// hermetic and never consult the host's actual container runtime (§11.4.50).

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

// getwdForTest is the os.Getwd seam the not-cwd assertions use.
func getwdForTest() (string, error) { return os.Getwd() }

// stubWeightsDirExec replaces the runtime-query seam and the runtime list for
// one test, restoring both on cleanup.
func stubWeightsDirExec(t *testing.T, runtimes []string, execFn func(runtime string, args ...string) (string, error)) {
	t.Helper()
	origExec, origRuntimes := weightsDirExec, containerRuntimes
	t.Cleanup(func() {
		weightsDirExec = origExec
		containerRuntimes = origRuntimes
	})
	weightsDirExec = func(_ context.Context, runtime string, args ...string) (string, error) {
		return execFn(runtime, args...)
	}
	containerRuntimes = runtimes
}

// stubLookPath replaces the runtime-on-PATH probe for one test.
func stubLookPath(t *testing.T, fn func(string) (string, error)) {
	t.Helper()
	orig := lookPath
	t.Cleanup(func() { lookPath = orig })
	lookPath = fn
}

// lookPathPresent returns a probe that reports every runtime as on PATH.
func lookPathPresent(string) (string, error) { return "/usr/bin/runtime", nil }

// notTheCwd asserts the resolver did not fall back to the working directory.
func notTheCwd(t *testing.T, dir string) {
	t.Helper()
	wd, err := getwdForTest()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if dir == wd {
		t.Fatalf("resolver returned the working directory %q — the EX-28 fallback it exists to prevent", dir)
	}
}

func TestResolveWeightsDir_EnvOverrideWins(t *testing.T) {
	t.Setenv("TEST_LANE_WEIGHTS_DIR", "/srv/operator-chosen-weights")
	stubWeightsDirExec(t, []string{"podman"}, func(runtime string, args ...string) (string, error) {
		t.Fatalf("runtime %q must not be queried when the env override is set (args=%v)", runtime, args)
		return "", nil
	})

	dir, err := ResolveWeightsDir(context.Background(), "TEST_LANE_WEIGHTS_DIR", "some-volume")
	if err != nil {
		t.Fatalf("ResolveWeightsDir: %v", err)
	}
	if dir != "/srv/operator-chosen-weights" {
		t.Fatalf("env override not honoured: got %q", dir)
	}
	notTheCwd(t, dir)
}

func TestResolveWeightsDir_ExistingVolumeMountpoint(t *testing.T) {
	stubLookPath(t, lookPathPresent)
	stubWeightsDirExec(t, []string{"podman"}, func(runtime string, args ...string) (string, error) {
		if len(args) == 4 && args[0] == "volume" && args[1] == "inspect" {
			if args[2] != "helixllm-videogen-cache" {
				t.Fatalf("queried the wrong volume: %v", args)
			}
			if args[3] != "--format={{.Mountpoint}}" {
				t.Fatalf("unexpected inspect format: %v", args)
			}
			return "/home/op/.local/share/containers/storage/volumes/helixllm-videogen-cache/_data", nil
		}
		t.Fatalf("unexpected query: %v", args)
		return "", nil
	})

	dir, err := ResolveWeightsDir(context.Background(), "TEST_LANE_WEIGHTS_DIR_UNSET", "helixllm-videogen-cache")
	if err != nil {
		t.Fatalf("ResolveWeightsDir: %v", err)
	}
	if !strings.HasSuffix(dir, "helixllm-videogen-cache/_data") {
		t.Fatalf("expected the volume mountpoint, got %q", dir)
	}
	notTheCwd(t, dir)
}

func TestResolveWeightsDir_GraphRootBeforeFirstAcquisition(t *testing.T) {
	// The volume does not exist yet (inspect fails); the runtime's storage
	// root is the honest answer — the named volume will be created on that
	// filesystem.
	stubLookPath(t, lookPathPresent)
	stubWeightsDirExec(t, []string{"podman"}, func(runtime string, args ...string) (string, error) {
		switch {
		case len(args) == 4 && args[0] == "volume":
			return "", errors.New("volume does not exist yet")
		case len(args) == 2 && args[0] == "info" && args[1] == "--format={{.Store.GraphRoot}}":
			return "/home/op/.local/share/containers/storage", nil
		}
		t.Fatalf("unexpected query: %v", args)
		return "", nil
	})

	dir, err := ResolveWeightsDir(context.Background(), "TEST_LANE_WEIGHTS_DIR_UNSET", "helixllm-videogen-cache")
	if err != nil {
		t.Fatalf("ResolveWeightsDir: %v", err)
	}
	if dir != "/home/op/.local/share/containers/storage" {
		t.Fatalf("expected the runtime storage root, got %q", dir)
	}
	notTheCwd(t, dir)
}

func TestResolveWeightsDir_DockerStorageRootQuery(t *testing.T) {
	// docker's info output carries the root under a different field than
	// podman's; the query must ask docker for {{.DockerRootDir}}.
	stubLookPath(t, lookPathPresent)
	stubWeightsDirExec(t, []string{"docker"}, func(runtime string, args ...string) (string, error) {
		if len(args) == 4 && args[0] == "volume" {
			return "", errors.New("volume does not exist yet")
		}
		if len(args) == 2 && args[0] == "info" && args[1] == "--format={{.DockerRootDir}}" {
			return "/var/lib/docker", nil
		}
		t.Fatalf("unexpected query: %v", args)
		return "", nil
	})

	dir, err := ResolveWeightsDir(context.Background(), "TEST_LANE_WEIGHTS_DIR_UNSET", "v")
	if err != nil {
		t.Fatalf("ResolveWeightsDir: %v", err)
	}
	if dir != "/var/lib/docker" {
		t.Fatalf("expected the docker storage root, got %q", dir)
	}
	notTheCwd(t, dir)
}

func TestResolveWeightsDir_EmptyMountpointFallsThroughToGraphRoot(t *testing.T) {
	stubLookPath(t, lookPathPresent)
	stubWeightsDirExec(t, []string{"podman"}, func(runtime string, args ...string) (string, error) {
		if args[0] == "volume" {
			return "  \n", nil // runtime answered with nothing usable
		}
		return "/var/lib/docker", nil
	})

	dir, err := ResolveWeightsDir(context.Background(), "TEST_LANE_WEIGHTS_DIR_UNSET", "v")
	if err != nil {
		t.Fatalf("ResolveWeightsDir: %v", err)
	}
	if dir != "/var/lib/docker" {
		t.Fatalf("expected the storage root after an empty mountpoint, got %q", dir)
	}
	notTheCwd(t, dir)
}

func TestResolveWeightsDir_NoRuntimeRefusesAndNamesTheRemedy(t *testing.T) {
	stubWeightsDirExec(t, []string{"podman", "docker"}, func(runtime string, args ...string) (string, error) {
		t.Fatalf("no runtime should be queried when none is on PATH")
		return "", nil
	})
	stubLookPath(t, func(string) (string, error) { return "", errors.New("not on PATH") })
	t.Setenv("TEST_LANE_WEIGHTS_DIR", "") // force-unset: the runtime path must be taken

	dir, err := ResolveWeightsDir(context.Background(), "TEST_LANE_WEIGHTS_DIR", "helixllm-videogen-cache")
	if err == nil {
		t.Fatalf("expected a refusal without a container runtime, got dir %q", dir)
	}
	if dir != "" {
		t.Fatalf("a refused resolution must not return a directory: %q", dir)
	}
	if !strings.Contains(err.Error(), "TEST_LANE_WEIGHTS_DIR") {
		t.Fatalf("the refusal must name the env override as the remedy: %v", err)
	}
	notTheCwd(t, dir)
}

func TestResolveWeightsDir_RuntimePresentButUnqueryableRefuses(t *testing.T) {
	// The runtime exists yet every query fails (daemon down): refuse, never
	// guess. This is the case that used to be answered by os.Getwd().
	stubLookPath(t, lookPathPresent)
	stubWeightsDirExec(t, []string{"podman"}, func(runtime string, args ...string) (string, error) {
		return "", errors.New("daemon unreachable")
	})
	t.Setenv("TEST_LANE_WEIGHTS_DIR", "")

	dir, err := ResolveWeightsDir(context.Background(), "TEST_LANE_WEIGHTS_DIR", "helixllm-videogen-cache")
	if err == nil {
		t.Fatalf("expected a refusal when the runtime cannot answer, got dir %q", dir)
	}
	notTheCwd(t, dir)
}

func TestResolveWeightsDir_DockerFallbackAfterPodmanAbsent(t *testing.T) {
	// podman is not on PATH; docker is asked instead and its answer wins.
	stubLookPath(t, func(name string) (string, error) {
		if name == "docker" {
			return "/usr/bin/docker", nil
		}
		return "", errors.New("not on PATH")
	})
	stubWeightsDirExec(t, []string{"podman", "docker"}, func(runtime string, args ...string) (string, error) {
		if runtime != "docker" {
			t.Fatalf("podman must be skipped when absent from PATH; queried %q", runtime)
		}
		if len(args) == 4 && args[0] == "volume" {
			return "/var/lib/docker/volumes/helixllm-imagegen-cache/_data", nil
		}
		t.Fatalf("unexpected query: %v", args)
		return "", nil
	})

	dir, err := ResolveWeightsDir(context.Background(), "TEST_LANE_WEIGHTS_DIR_UNSET", "helixllm-imagegen-cache")
	if err != nil {
		t.Fatalf("ResolveWeightsDir: %v", err)
	}
	if !strings.HasSuffix(dir, "helixllm-imagegen-cache/_data") {
		t.Fatalf("expected the docker volume mountpoint, got %q", dir)
	}
	notTheCwd(t, dir)
}

func TestResolveWeightsDir_RuntimeOrderIsPodmanFirst(t *testing.T) {
	// Rootless podman is the mandated runtime (§11.4.161); it must be asked
	// before docker so a host with both gets the podman answer.
	if containerRuntimes[0] != "podman" || containerRuntimes[1] != "docker" {
		t.Fatalf("runtime order changed: %v", containerRuntimes)
	}
}

func TestResolveWeightsDir_UsesContextDeadline(t *testing.T) {
	// The probe must not outlive the caller's context: a cancelled context
	// surfaces as a refusal, not a hang.
	stubLookPath(t, lookPathPresent)
	stubWeightsDirExec(t, []string{"podman"}, func(runtime string, args ...string) (string, error) {
		return "", errors.New("unreachable")
	})
	t.Setenv("TEST_LANE_WEIGHTS_DIR", "")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := ResolveWeightsDir(ctx, "TEST_LANE_WEIGHTS_DIR", "v")
	if err == nil {
		t.Fatal("expected a refusal under a cancelled context")
	}
}

func TestResolveWeightsDir_TimeoutBudgetIsBounded(t *testing.T) {
	// Regression shape from OPEN-34's neighbour: every runtime probe is
	// bounded so a wedged daemon cannot stall the decision.
	if weightsDirProbeTimeout <= 0 || weightsDirProbeTimeout > 30*time.Second {
		t.Fatalf("probe timeout %v is unbounded or implausibly large", weightsDirProbeTimeout)
	}
}
