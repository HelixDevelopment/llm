// Command imagegen-boot is the on-demand boot + health + single-owner-teardown
// harness for the HelixLLM Phase-4 GPU image-generation service (FLUX + Nunchaku
// NVFP4 on the host RTX 5090). It is the ready-to-run scaffold that makes the
// eventual runtime proof a fast-follow once the operator authorizes a
// coder-pause window (§11.4.122); it does NOT itself run a GPU generation.
//
// The load-bearing discipline (design scratchpad/design_gpu_generative*.md):
//
//  1. BEFORE booting the container, the NVFP4 VRAM footprint MUST be admitted
//     by the vrambroker (ClassImage, BURST, single-owner §11.4.119) — NEVER a
//     raw VRAM grab. Admission is gated on the MEASURED free VRAM from
//     nvidia-smi + a 2 GiB headroom (broker.admit, fail-closed §11.4.6).
//     * granted  -> co-resident fast path (coder stays live) -> boot.
//     * ErrBudgetExceeded -> the NVFP4 min-mem config (~6 GB peak) does not
//     fit alongside the live coder RIGHT NOW; the coder-pause path is
//     required. This harness DOES NOT pause the live coder autonomously
//     (pausing a shipped capability is operator-gated, §11.4.122/§11.4.101)
//     — it reports BLOCKED and exits, leaving the coder untouched.
//     * ErrBurstInUse   -> another image/video burst owns the card (queue).
//     * ErrBudgetUnavailable -> nvidia-smi unreadable -> refuse fail-closed.
//  2. Boot the service on its OWN port (:18442) through the containers
//     submodule compose.Orchestrator (§11.4.76), rootless podman (§11.4.161).
//  3. Health-poll /health (containers pkg/health) until the shim answers.
//  4. Single-owner teardown (compose down) + Release the burst lease — the
//     coder (:18434) is never touched.
//
// WHICH MODEL RUNS IS MEASURED, NOT CONFIGURED (FR-056)
//
// This harness does not have a default model and cannot be told which model to
// run. Every boot measures this host, joins the measurement against the
// recorded catalogue under the declared usage, and serves an option the host
// was proven able to run. See modelchoice.go for the decision and the three
// distinct reasons a candidate can be withheld.
//
//   - IMAGEGEN_MODEL and IMAGEGEN_PRECISION are OUTPUTS of that decision,
//     written here for compose to interpolate. They are no longer inputs: a
//     value found in the environment is reported and overwritten, because a
//     configured name would defeat the measurement. Previously the model was
//     whatever the ambient environment happened to hold — nothing measured the
//     host, so nothing could refuse a model this card cannot run.
//   - The VRAM figure admitted by the broker comes from the CHOSEN option's
//     recorded requirement. IMAGEGEN_NEED_BYTES is no longer honoured: a static
//     footprint implies a model, which is the static selection FR-056 forbids.
//
// Subcommands:
//
//	plan   [--pin id[:variant]]     measure + decide + report; boots nothing
//	admit-check                     broker-only VRAM admission verdict (no boot)
//	boot   <compose-file> <project> measure -> choose -> admit -> up -> health
//	down   <compose-file> <project> single-owner teardown (compose down)
//	status <compose-file> <project> compose service status
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"digital.vasic.containers/pkg/compose"
	"digital.vasic.containers/pkg/health"

	"github.com/HelixDevelopment/HelixLLM/internal/catalogue"
	"github.com/HelixDevelopment/HelixLLM/internal/laneboot"
	"github.com/HelixDevelopment/HelixLLM/internal/vrambroker"
)

const (
	service    = "imagegen"
	healthPort = "18442" // OWN port — coder :18434 + siblings :18435-18441 untouched

	// family is the capability this lane serves. It selects WITHIN the
	// catalogue; it never names a model.
	family = catalogue.FamilyImageGeneration

	// forbidKey is the operator's forbid-list. Forbidding can only remove an
	// option the measurement offered, never add one it did not.
	forbidKey = "IMAGEGEN_FORBID_MODELS"

	// modelKey and precisionKey are compose interpolation OUTPUTS, written from
	// the measured decision.
	modelKey     = "IMAGEGEN_MODEL"
	precisionKey = "IMAGEGEN_PRECISION"
)

// GiB is one gibibyte in bytes.
const GiB int64 = 1024 * 1024 * 1024

// chooseModel measures the host and decides which image-generation model this
// host can actually serve.
//
// Unlike the vision lane there are no local weight files to locate: the runtime
// pulls the chosen model's weights by repository. What must still be resolved
// is that the chosen build is one this runtime can serve at all — an entry
// whose precision the runtime does not implement is reported, not coerced.
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
		precision, perr := precisionFor(option)
		if perr != nil {
			unservable = append(unservable, fmt.Sprintf("%s (%v)", option.Identity, perr))
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
			Precision:  precision,
		}, nil
	}

	return choice{}, laneboot.ExitErr(exitNotServable,
		"CANNOT-CHOOSE: this host was measured and can serve %d %s model(s), but none of them is servable "+
			"by this runtime:\n  %s\n"+
			"  No model is started: falling back to whatever default the runtime carries would be a model "+
			"nobody chose.\n  Remedy: extend the runtime to serve one of the builds above, or record a "+
			"servable build in the catalogue.",
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
	fmt.Printf("  weights:  %s (precision %s)\n", c.Repository, c.Precision)
}

// applyChoice writes the decision into the environment compose interpolates.
//
// These variables are OUTPUTS. A value already present in the environment named
// a model that no measurement chose, so it is reported and overwritten rather
// than honoured.
func applyChoice(c choice) {
	for key, value := range map[string]string{
		modelKey:     c.Repository,
		precisionKey: c.Precision,
	} {
		if existing := os.Getenv(key); existing != "" && existing != value {
			fmt.Printf("IGNORED-CONFIG: %s=%q named a model that no measurement chose; "+
				"overwritten with the measured choice %q (FR-056).\n", key, existing, value)
		}
		os.Setenv(key, value)
	}
	if legacy := os.Getenv("IMAGEGEN_NEED_BYTES"); legacy != "" {
		fmt.Printf("IGNORED-CONFIG: IMAGEGEN_NEED_BYTES=%q is no longer honoured — it implied a model. "+
			"The admitted figure comes from the chosen option's recorded requirement (FR-056).\n", legacy)
	}
}

// needBytesFor is the VRAM footprint to admit for the chosen option: its
// recorded memory requirement. The broker adds its own headroom on top.
func needBytesFor(c choice) int64 {
	return int64(c.Option.Cost.MemoryRequiredBytes)
}

// cmdPlan measures, decides and reports — and boots nothing.
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
		fatal("usage: imagegen-boot <plan|admit-check|boot|down|status> [compose-file] [project] [--pin id[:variant]]")
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

// admit acquires a burst (ClassImage) lease for the chosen model's footprint,
// returning the lease or a classified reason. It NEVER pauses the coder — an
// ErrBudgetExceeded is surfaced as a BLOCKED verdict (coder-pause is operator
// gated, §11.4.122).
func admit(ctx context.Context, need int64) (*vrambroker.Lease, error) {
	broker := vrambroker.New() // real nvidia-smi-backed admission (§11.4.6 fail-closed)
	total, used, free := broker.Budget()
	fmt.Printf("VRAM budget (nvidia-smi): total=%dMiB used=%dMiB free=%dMiB need=%dMiB headroom=%dMiB\n",
		total/(1024*1024), used/(1024*1024), free/(1024*1024),
		need/(1024*1024), vrambroker.HeadroomBytes/(1024*1024))
	lease, err := broker.Acquire(ctx, vrambroker.ClassImage, need)
	return lease, err
}

// classifyAdmit prints a human verdict for an Acquire error and returns an exit
// code (0 = admitted, non-zero = not admitted / blocked).
func classifyAdmit(err error) int {
	switch {
	case err == nil:
		fmt.Println("ADMIT-OK: chosen model's footprint admitted co-resident (coder stays live) — fast path")
		return 0
	case errors.Is(err, vrambroker.ErrBudgetExceeded):
		fmt.Println("BLOCKED: ErrBudgetExceeded — the chosen model does not fit alongside the live coder right now.")
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
		Services: []string{"imagegen"},
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

	// (3) boot on :18442 through the containers submodule orchestrator.
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
	fmt.Printf("UP-OK: %s imagegen via containers submodule orchestrator (:%s) serving %s\n",
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
		fmt.Printf("DOWN-OK: %s imagegen torn down (single-owner cleanup, coder untouched)\n", p.Name)
	}

	if !ok {
		fmt.Println("BLOCKED: imagegen service never became healthy on :" + healthPort)
		os.Exit(4)
	}
	fmt.Println("BOOT-HEALTH-OK: imagegen /health answered. Generation + VRAM calibration is the")
	fmt.Println("PENDING runtime-proof step (operator-authorized coder-pause + first-run calibration).")
}

// pollHealth probes the shim /health via the containers pkg/health HTTP checker
// (§11.4.76 primitive) until it answers or the deadline passes.
func pollHealth(ctx context.Context, budget time.Duration) bool {
	target := health.HealthTarget{
		Name:    "imagegen",
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
			fmt.Printf("HEALTH-OK: imagegen /health after %d polls (status=%s)\n", n, res.Details["status_code"])
			return true
		}
		time.Sleep(3 * time.Second)
	}
	fmt.Printf("HEALTH-TIMEOUT: imagegen /health did not answer within %s (%d polls)\n", budget, n)
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
	fmt.Printf("DOWN-OK: %s imagegen (single-owner cleanup, coder untouched)\n", p.Name)
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
