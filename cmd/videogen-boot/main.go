// Command videogen-boot is the on-demand boot + health + single-owner-teardown
// harness for the HelixLLM Phase-4 GPU video-generation service (WAN 2.2 /
// LTX-Video on the host accelerator). It boots the REAL diffusion service in
// services/videogen; it does not itself run a generation.
//
// WHICH MODEL RUNS IS MEASURED, NOT CONFIGURED (FR-056)
//
// This harness has no default model and cannot be told which model to run.
// Every boot measures this host, joins the measurement against the recorded
// catalogue under the declared usage, and serves an option the host was proven
// able to run. See modelchoice.go for the decision and the three distinct
// reasons a candidate can be withheld (FR-055).
//
//   - VIDEOGEN_MODEL, VIDEOGEN_BACKEND, VIDEOGEN_PRECISION, VIDEOGEN_SIZE,
//     VIDEOGEN_NUM_FRAMES and VIDEOGEN_FPS are OUTPUTS of that decision,
//     written here for compose to interpolate. They are no longer inputs: a
//     value found in the environment is reported and overwritten, because a
//     configured name — or a configured clip shape whose memory cost nobody
//     measured — would defeat the measurement.
//   - The VRAM figure admitted by the broker comes from the CHOSEN option's
//     recorded requirement. VIDEOGEN_NEED_BYTES is no longer honoured: it was a
//     STATIC 10 GiB standing in for a per-model figure, which is exactly the
//     static selection FR-056 forbids. A single fixed number cannot be right
//     for three models whose recorded requirements differ.
//   - VIDEOGEN_CPU_OFFLOAD, VIDEOGEN_MAX_STEPS, VIDEOGEN_HOST_PORT,
//     VIDEOGEN_MEM_LIMIT and VIDEOGEN_SHM_SIZE remain INPUTS. They say HOW and
//     WHERE the service runs; none of them names a model.
//
// NO PROCESSOR-VIABLE OPTION EXISTS for this family, by design. Every recorded
// video entry requires an accelerator, so a host without one is offered
// NOTHING and told what it lacks — never handed a fallback that would start and
// then be unusable. That refusal is the correct behaviour.
//
// The rest of the discipline is unchanged (design scratchpad/
// design_gpu_generative*.md):
//
//  1. BEFORE booting the container, the CHOSEN model's footprint MUST be
//     admitted by the vrambroker (ClassVideo, BURST, single-owner §11.4.119) —
//     NEVER a raw VRAM grab. Admission is gated on the MEASURED free VRAM from
//     nvidia-smi plus headroom (fail-closed §11.4.6).
//     * granted  -> co-resident fast path (coder stays live) -> boot.
//     * ErrBudgetExceeded -> the chosen config does not fit alongside the live
//     coder RIGHT NOW; the coder-pause path is required. This harness DOES NOT
//     pause the live coder autonomously (pausing a shipped capability is
//     operator-gated, §11.4.122/§11.4.101) — it reports BLOCKED and exits,
//     leaving the coder untouched.
//     * ErrBurstInUse   -> another image/video burst owns the card (queue).
//     * ErrBudgetUnavailable -> nvidia-smi unreadable -> refuse fail-closed.
//  2. Boot the service on its OWN port (:18443) through the containers
//     submodule compose.Orchestrator (§11.4.76), rootless podman (§11.4.161).
//     Distinct from the coder (:18434) and the image-gen sibling (:18442).
//  3. Health-poll /health (containers pkg/health) until the shim answers.
//  4. Single-owner teardown (compose down) + Release the burst lease — the
//     coder (:18434) is never touched.
//
// Subcommands:
//
//	plan   [--pin id[:variant]]     measure + decide + report; boots nothing
//	admit-check                     measure -> choose -> broker verdict (no boot)
//	boot   <compose-file> <project> measure -> choose -> admit -> up -> health
//	down   <compose-file> <project> single-owner teardown (compose down)
//	status <compose-file> <project> compose service status
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"digital.vasic.containers/pkg/compose"
	"digital.vasic.containers/pkg/health"

	"github.com/HelixDevelopment/HelixLLM/internal/catalogue"
	"github.com/HelixDevelopment/HelixLLM/internal/laneboot"
	"github.com/HelixDevelopment/HelixLLM/internal/vrambroker"
)

const (
	service    = "videogen"
	healthPort = "18443" // OWN port — coder :18434 + image-gen :18442 untouched

	// family is the capability this lane serves. It selects WITHIN the
	// catalogue; it never names a model.
	family = catalogue.FamilyVideoGeneration

	// forbidKey is the operator's forbid-list. Forbidding can only remove an
	// option the measurement offered, never add one it did not.
	forbidKey = "VIDEOGEN_FORBID_MODELS"

	// These are compose interpolation OUTPUTS, written from the measured
	// decision. Nothing reads them as input.
	modelKey     = "VIDEOGEN_MODEL"
	backendKey   = "VIDEOGEN_BACKEND"
	precisionKey = "VIDEOGEN_PRECISION"
	sizeKey      = "VIDEOGEN_SIZE"
	framesKey    = "VIDEOGEN_NUM_FRAMES"
	fpsKey       = "VIDEOGEN_FPS"

	// legacyNeedKey named a VRAM figure, and a figure implies a model. It is
	// reported and ignored rather than silently dropped.
	legacyNeedKey = "VIDEOGEN_NEED_BYTES"
)

// chooseModel measures the host and decides which video-generation model this
// host can actually serve.
//
// The order is the whole point: measure, then choose, then ask whether this
// runtime can serve the chosen build at all. Every failure path refuses; none
// substitutes a default model.
func chooseModel(ctx context.Context) (choice, error) {
	pin, err := laneboot.ParsePin(os.Args[1:])
	if err != nil {
		return choice{}, laneboot.ExitErr(laneboot.ExitNoOptionOffered, "CANNOT-CHOOSE: %v", err)
	}

	dir, err := weightsDir(ctx)
	if err != nil {
		return choice{}, err
	}

	offered, loaded, profile, purpose, err := laneboot.Decide(ctx, family, dir, pin, forbidKey)
	if err != nil {
		return choice{}, err
	}

	var unservable []string
	// offered arrives ordered cheapest-admissible-first from selection, and is
	// taken in that order: the first option this runtime can actually serve wins.
	// The order is not re-decided here — one rule, in one place, so the lanes
	// cannot drift apart from each other or from the Python gate.
	for _, option := range offered {
		backend, berr := backendFor(option)
		if berr != nil {
			unservable = append(unservable, fmt.Sprintf("%s (%v)", option.Identity, berr))
			continue
		}
		precision, perr := precisionFor(option)
		if perr != nil {
			unservable = append(unservable, fmt.Sprintf("%s (%v)", option.Identity, perr))
			continue
		}
		shape, serr := videoShapeFor(option)
		if serr != nil {
			unservable = append(unservable, fmt.Sprintf("%s (%v)", option.Identity, serr))
			continue
		}
		entry, ok := laneboot.EntryFor(loaded, option)
		if !ok {
			unservable = append(unservable, fmt.Sprintf("%s (no catalogue entry to take a weight source from)", option.Identity))
			continue
		}
		repo, rerr := repositoryFor(entry)
		if rerr != nil {
			unservable = append(unservable, fmt.Sprintf("%s (%v)", option.Identity, rerr))
			continue
		}
		return choice{
			Option:     option,
			Entry:      entry,
			Profile:    profile,
			Usage:      purpose,
			Repository: repo,
			Backend:    backend,
			Precision:  precision,
			Shape:      shape,
		}, nil
	}

	return choice{}, laneboot.ExitErr(exitNotServable,
		"CANNOT-CHOOSE: this host was measured and can serve %d %s model(s), but none of them is servable "+
			"by this runtime:\n  %s\n"+
			"  No model is started: falling back to whatever default the runtime carries would be a model "+
			"nobody chose, and its footprint would be one nobody measured.\n"+
			"  Remedy: record a servable build in the catalogue, or extend the runtime to serve one of the "+
			"builds above.",
		len(offered), family, joinLines(unservable))
}

// reportChoice prints the decision and the measurement it rests on.
func reportChoice(c choice) {
	fmt.Printf("CHOSEN %s — decided from the measured host %q, not from configuration.\n",
		c.Option.Identity, c.Option.HostIdentity)
	fmt.Printf("  requires: memory=%dMiB storage=%dMiB accelerator=%t\n",
		c.Option.Cost.MemoryRequiredBytes/(1024*1024),
		c.Option.Cost.StorageRequiredBytes/(1024*1024),
		c.Option.Cost.RequiresAccelerator)
	fmt.Printf("  leaves:   memory=%dMiB (%.1f%% of total) storage=%dMiB\n",
		c.Option.Headroom.MemoryRemainingBytes/(1024*1024),
		c.Option.Headroom.MemoryRemainingFraction*100,
		c.Option.Headroom.StorageRemainingBytes/(1024*1024))
	fmt.Printf("  licence:  %s permits the declared usage %q\n", c.Option.Terms.LicenseID, c.Usage)
	fmt.Printf("  weights:  %s (backend %s, precision %s)\n", c.Repository, c.Backend, c.Precision)
	fmt.Printf("  clip:     %s, %d frames @ %d fps (%.2fs) — the shape this option was selected AT\n",
		c.Shape.Size, c.Shape.NumFrames, c.Shape.FPS, c.Option.Expected.Video.DurationSeconds())
}

// applyChoice writes the decision into the environment compose interpolates.
//
// These variables are OUTPUTS. A value already present in the environment named
// a model — or a clip shape — that no measurement chose, so it is reported and
// overwritten rather than honoured.
func applyChoice(c choice) {
	for key, value := range map[string]string{
		modelKey:     c.Repository,
		backendKey:   c.Backend,
		precisionKey: c.Precision,
		sizeKey:      c.Shape.Size,
		framesKey:    strconv.Itoa(c.Shape.NumFrames),
		fpsKey:       strconv.Itoa(c.Shape.FPS),
	} {
		if existing := os.Getenv(key); existing != "" && existing != value {
			fmt.Printf("IGNORED-CONFIG: %s=%q described a model that no measurement chose; "+
				"overwritten with the measured choice %q (FR-056).\n", key, existing, value)
		}
		os.Setenv(key, value)
	}
	if legacy := os.Getenv(legacyNeedKey); legacy != "" {
		fmt.Printf("IGNORED-CONFIG: %s=%q is no longer honoured — a static VRAM figure implied a model. "+
			"The admitted figure comes from the chosen option's recorded requirement (FR-056).\n",
			legacyNeedKey, legacy)
	}
}

// needBytesFor is the VRAM footprint to admit for the chosen option: its
// recorded memory requirement. The broker adds its own headroom on top.
func needBytesFor(c choice) int64 {
	return int64(c.Option.Cost.MemoryRequiredBytes)
}

// cmdPlan measures, decides and reports — and boots nothing. It is the honest
// way to see which model this host would serve, and why the others were not
// offered, without touching the card.
func cmdPlan() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	c, err := chooseModel(ctx)
	if err != nil {
		fmt.Println(err)
		os.Exit(laneboot.ExitCodeFor(err))
	}
	reportChoice(c)
	fmt.Printf("PLAN-OK: this host would serve %s (nothing was booted).\n", c.Option.Identity)
}

// joinLines renders a list one per indented line, for refusals that name every
// option they considered.
func joinLines(items []string) string {
	out := ""
	for i, s := range items {
		if i > 0 {
			out += "\n  "
		}
		out += s
	}
	return out
}

func main() {
	if len(os.Args) < 2 {
		fatal("usage: videogen-boot <plan|admit-check|boot|down|status> [compose-file] [project] [--pin id[:variant]]")
	}
	switch os.Args[1] {
	case "plan":
		cmdPlan()
	case "admit-check":
		cmdAdmitCheck()
	case "boot":
		cmdBoot()
	case "down":
		cmdDown()
	case "status":
		cmdStatus()
	default:
		fatal("unknown subcommand: %s", os.Args[1])
	}
}

// admit acquires a burst (ClassVideo) lease for the CHOSEN model's footprint,
// returning the lease or a classified reason. It NEVER pauses the coder — an
// ErrBudgetExceeded is surfaced as a BLOCKED verdict (coder-pause is operator
// gated, §11.4.122).
func admit(ctx context.Context, need int64) (*vrambroker.Lease, error) {
	broker := vrambroker.New() // real nvidia-smi-backed admission (§11.4.6 fail-closed)
	total, used, free := broker.Budget()
	fmt.Printf("VRAM budget (nvidia-smi): total=%dMiB used=%dMiB free=%dMiB need=%dMiB headroom=%dMiB\n",
		total/(1024*1024), used/(1024*1024), free/(1024*1024),
		need/(1024*1024), vrambroker.HeadroomBytes/(1024*1024))
	lease, err := broker.Acquire(ctx, vrambroker.ClassVideo, need)
	return lease, err
}

// classifyAdmit prints a human verdict for an Acquire error and returns an exit
// code (0 = admitted, non-zero = not admitted / blocked).
func classifyAdmit(err error) int {
	switch {
	case err == nil:
		fmt.Println("ADMIT-OK: video footprint admitted co-resident (coder stays live) — fast path")
		return 0
	case errors.Is(err, vrambroker.ErrBudgetExceeded):
		fmt.Println("BLOCKED: ErrBudgetExceeded — the selected no-pause config does not fit alongside the live coder right now.")
		fmt.Println("         The coder-pause path is required (operator-gated §11.4.122); this harness does NOT")
		fmt.Println("         pause the live coder autonomously. Coder untouched.")
		return 10
	case errors.Is(err, vrambroker.ErrBurstInUse):
		fmt.Println("BLOCKED: ErrBurstInUse — another image/video burst owns the card (single-owner §11.4.119). Queue.")
		return 11
	case errors.Is(err, vrambroker.ErrBudgetUnavailable):
		fmt.Println("BLOCKED: ErrBudgetUnavailable — nvidia-smi unreadable; refusing fail-closed (§11.4.6). No boot.")
		return 12
	case errors.Is(err, vrambroker.ErrThermalUnsafe):
		fmt.Println("BLOCKED: ErrThermalUnsafe — card outside safe thermal/power envelope (§11.4.133). No boot.")
		return 13
	default:
		fmt.Printf("BLOCKED: admission failed: %v\n", err)
		return 14
	}
}

// cmdAdmitCheck tests the admission gate for the model this host would actually
// serve. It measures first, because the footprint to admit is the chosen
// model's — there is no fixed figure to test against.
func cmdAdmitCheck() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	c, cerr := chooseModel(ctx)
	if cerr != nil {
		fmt.Println(cerr)
		os.Exit(laneboot.ExitCodeFor(cerr))
	}
	reportChoice(c)

	lease, err := admit(ctx, needBytesFor(c))
	code := classifyAdmit(err)
	if lease != nil {
		// admit-check only tests the gate — release immediately (§11.4.119).
		lease.Release()
	}
	os.Exit(code)
}

func project() compose.ComposeProject {
	if len(os.Args) < 4 {
		fatal("need <compose-file> <project>")
	}
	return compose.ComposeProject{
		Name:     os.Args[3],
		File:     os.Args[2],
		Services: []string{service},
	}
}

func cmdBoot() {
	p := project()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	// (1) WHICH model — measured, never configured. A refusal here exits
	// non-zero with the reason; no default model stands in for a measurement.
	c, cerr := chooseModel(ctx)
	if cerr != nil {
		fmt.Println(cerr)
		os.Exit(laneboot.ExitCodeFor(cerr))
	}
	reportChoice(c)
	applyChoice(c)

	// (2) admit the CHOSEN model's footprint BEFORE boot (§11.4.119 / broker).
	lease, err := admit(ctx, needBytesFor(c))
	if code := classifyAdmit(err); code != 0 {
		os.Exit(code)
	}
	defer lease.Release() // single-owner slot freed on exit no matter what

	// (3) boot on :18443 through the containers submodule orchestrator.
	orch, oerr := compose.NewDefaultOrchestrator(".", nil)
	if oerr != nil {
		fatal("orchestrator: %v", oerr)
	}
	if err := orch.Up(ctx, p,
		compose.WithUpDetach(true),
		compose.WithRemoveOrphans(true),
	); err != nil {
		fatal("compose up: %v", err)
	}
	fmt.Printf("UP-OK: %s videogen via containers submodule orchestrator (:%s) serving %s\n",
		p.Name, healthPort, c.Option.Identity)

	// (4) health poll.
	ok := pollHealth(ctx, 5*time.Minute)

	// (5) single-owner teardown — coder untouched.
	dctx, dcancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer dcancel()
	if derr := orch.Down(dctx, p,
		compose.WithDownRemoveVolumes(false), // keep the HF model cache
		compose.WithDownRemoveOrphans(true),
	); derr != nil {
		fmt.Printf("WARN: compose down: %v\n", derr)
	} else {
		fmt.Printf("DOWN-OK: %s videogen torn down (single-owner cleanup, coder untouched)\n", p.Name)
	}

	if !ok {
		fmt.Println("BLOCKED: videogen service never became healthy on :" + healthPort)
		os.Exit(4)
	}
	fmt.Printf("BOOT-HEALTH-OK: videogen /health answered for %s. Generation + first-run VRAM\n",
		c.Option.Identity)
	fmt.Println("calibration remain the PENDING runtime-proof step (operator-authorized coder-pause).")
}

// pollHealth probes the shim /health via the containers pkg/health HTTP checker
// (§11.4.76 primitive) until it answers or the deadline passes.
func pollHealth(ctx context.Context, budget time.Duration) bool {
	target := health.HealthTarget{
		Name:    service,
		URL:     "http://localhost:" + healthPort + "/health",
		Type:    health.HealthHTTP,
		Timeout: 5 * time.Second,
	}
	deadline := time.Now().Add(budget)
	n := 0
	for time.Now().Before(deadline) {
		if ctx.Err() != nil {
			return false
		}
		n++
		res := health.CheckHTTP(ctx, target)
		if res.Healthy {
			fmt.Printf("HEALTH-OK: videogen /health after %d polls (status=%s)\n", n, res.Details["status_code"])
			return true
		}
		time.Sleep(3 * time.Second)
	}
	fmt.Printf("HEALTH-TIMEOUT: videogen /health did not answer within %s (%d polls)\n", budget, n)
	return false
}

func cmdDown() {
	p := project()
	orch, err := compose.NewDefaultOrchestrator(".", nil)
	if err != nil {
		fatal("orchestrator: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	if err := orch.Down(ctx, p,
		compose.WithDownRemoveVolumes(false),
		compose.WithDownRemoveOrphans(true),
	); err != nil {
		fatal("compose down: %v", err)
	}
	fmt.Printf("DOWN-OK: %s videogen (single-owner cleanup, coder untouched)\n", p.Name)
}

func cmdStatus() {
	p := project()
	orch, err := compose.NewDefaultOrchestrator(".", nil)
	if err != nil {
		fatal("orchestrator: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	sts, err := orch.Status(ctx, p)
	if err != nil {
		fatal("status: %v", err)
	}
	if len(sts) == 0 {
		fmt.Println("(no services reported)")
		return
	}
	for _, s := range sts {
		fmt.Printf("%s state=%s health=%s ports=%v exit=%d\n",
			s.Name, s.State, s.Health, s.Ports, s.ExitCode)
	}
}

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "ERROR: "+format+"\n", a...)
	os.Exit(2)
}
