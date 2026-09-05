package runtime

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/HelixDevelopment/HelixLLM/internal/catalogue"
	"github.com/HelixDevelopment/HelixLLM/internal/vrambroker"
)

// Process lifecycle for the disk-streaming runtime, under the same lease and
// health discipline the in-memory serving path already follows:
//
//	admit (vrambroker lease) → start → poll health within a budget → serve
//	                         → stop → release
//
// with the lease released on EVERY exit from that sequence, including the ones
// that fail. That ordering is the whole discipline. A process started before
// admission spends the capacity the gate exists to protect; a lease released
// before its process is gone hands out capacity that is still in use; a lease
// never released removes it from every other workload until the program exits.
//
// WHAT SATISFIES THE SEAM. The T070 deferral is lifted: colibri_process.go
// wires the adopted streaming runtime (Colibri) into this lifecycle —
// ColibriProcess implements Process, ColibriHealth implements HealthProbe,
// and NewColibriProcess is the config-presence gate the adoption was waiting
// for (a deployment that has not named a launcher binary is refused honestly,
// never faked). Nothing in this file imports the runtime, and the module's
// dependencies are unchanged: adoption is a launch-and-lifecycle integration,
// not a library one, exactly as colibri.go records. A containerised streaming
// runtime is brought up per §11.4.76 through the containers submodule
// (pkg/compose, pkg/health) rather than by hand — see deploy/compose.yaml's
// colibri service.

// Lease is a granted reservation against the host's admission budget, held for
// as long as a process is using that capacity.
//
// It is ONE seam for both serving paths. lease.go turns a selected option into
// a held reservation for the in-memory path and admits through this same gate;
// the streaming path admits through it too, which is what "the same lease
// discipline" means. A second pair of interfaces with the same shape would let
// the two paths drift into admitting against two different gates while both
// looked correct.
//
// It is an interface rather than *vrambroker.Lease so this lifecycle can be
// exercised without a card present. Taking the concrete type would make every
// test of the release paths require the one piece of hardware whose exhaustion
// those paths exist to prevent, so the paths would go untested on exactly the
// machines that run them.
type Lease interface {
	// Release returns the reservation. It must be idempotent: Close is reached
	// from a defer, a signal handler and an error path at once, and returning
	// capacity twice that was taken once is worse than not returning it.
	Release()
}

// Admitter is the host's admission gate. The production value is a
// vrambroker.Broker, adapted by BrokerAdmitter.
//
// What the streaming path does NOT share with lease.go's Acquire is the FIGURE
// it admits against. That function admits a selection.Option's recorded memory
// requirement — the right figure for a model held whole in memory, and
// deliberately "not a fraction of one". The streaming path holds only its
// resident working set while the remainder streams from disk, so it admits that
// figure instead, derived in PlanLaunch. Admitting the full requirement here
// would refuse exactly the hosts this path exists to serve.
type Admitter interface {
	Acquire(ctx context.Context, class vrambroker.Class, needBytes int64) (Lease, error)
}

// BrokerAdmitter adapts a vrambroker.Broker into the admission gate this
// package uses, so the real gate is wired here rather than described here.
func BrokerAdmitter(b vrambroker.Broker) Admitter { return brokerAdmitter{b: b} }

type brokerAdmitter struct{ b vrambroker.Broker }

func (a brokerAdmitter) Acquire(ctx context.Context, class vrambroker.Class, needBytes int64) (Lease, error) {
	lease, err := a.b.Acquire(ctx, class, needBytes)
	if err != nil {
		// Returned as a nil INTERFACE, deliberately: returning the typed nil
		// *vrambroker.Lease here would produce a non-nil Lease holding nothing,
		// and every "was anything granted?" check downstream would be wrong.
		//
		// The gate's own error travels out whole, so a caller can still tell an
		// over-budget refusal from a busy burst slot from an unreadable budget
		// and decide whether to queue, degrade or surface it.
		return nil, err
	}
	if lease == nil {
		return nil, fmt.Errorf("%w: class=%s need=%d", ErrNoReservation, class, needBytes)
	}
	return lease, nil
}

// Budget forwards the gate's live reading, so an admission refusal can record
// what the card actually had free. lease.go's Budgeter is satisfied by this.
func (a brokerAdmitter) Budget() (total, used, free int64) { return a.b.Budget() }

// Process is the streaming runtime's process, as this package needs to see it.
//
// This is the T070 seam, now satisfied by colibri_process.go's ColibriProcess.
// The interface stays: the lifecycle is tested against it without the runtime,
// and the deployment can still substitute any other Process implementation.
type Process interface {
	// Start brings the process up. It returns once the process is running, not
	// once it is ready — readiness is HealthProbe's question, because a
	// streaming runtime is running long before it can answer.
	Start(ctx context.Context) error
	// Stop tears it down. It is called on every failure path after a Start was
	// attempted, including one that failed, since a partial start can leave a
	// process behind.
	Stop(ctx context.Context) error
}

// HealthProbe reports whether the started process can serve yet.
//
// A bool rather than an error: the caller polls this, and "not ready yet" is
// the expected answer for most of a streaming load, not a fault. Distinguishing
// kinds of not-ready would invite treating one of them as fatal, and a slow
// load would then be reported as a broken runtime.
type HealthProbe interface {
	Healthy(ctx context.Context) bool
}

// Errors this file reports. They are separate because they have separate
// remedies, and a caller that cannot tell them apart cannot respond to them.
var (
	// ErrNotStreamingChoice: a choice for the in-memory path was handed to the
	// streaming launcher. Starting the streaming runtime for a model the
	// decision said to serve from memory is the D6 error — streaming as a
	// preference — committed at launch time instead of at choice time.
	ErrNotStreamingChoice = errors.New("runtime: launch plan requires a streaming choice")
	// ErrChoiceEntryMismatch: the choice and the entry describe different
	// models. The plan takes its path from one and its figures from the other,
	// so a mismatch would admit one model's memory for another model's process
	// and nothing later would catch it.
	ErrChoiceEntryMismatch = errors.New("runtime: launch plan's choice and entry are different models")
	// ErrNotHealthy: the process started and did not answer within the budget.
	// This is distinct from a failed start — the process ran — and the
	// distinction is the whole diagnosis.
	ErrNotHealthy = errors.New("runtime: streaming runtime did not become healthy within its budget")
	// ErrIncompletePlan: the launcher was not given what it needs to run the
	// discipline. It refuses rather than skipping a step: a launcher with no
	// admitter would start a process with no reservation, and a launcher with
	// no health probe would call anything it started ready.
	ErrIncompletePlan = errors.New("runtime: launcher is missing a component the lifecycle requires")
)

// LaunchPlan is everything needed to start one streaming model, derived from a
// choice that has already been made.
//
// It is a separate value from Choice because a choice is an answer about what
// CAN serve, and a plan is an instruction to spend resources. Deriving one from
// the other is the point at which the resident figure, the class and the
// trade-off are fixed, and having that be a named step means it can be checked.
type LaunchPlan struct {
	// Identity is the model this plan starts, for logs and for the session.
	Identity string
	// Class is the residency class the reservation is taken under.
	//
	// It is supplied by the caller, never inferred here. Which tier a streaming
	// model occupies is a deployment decision about how it shares a host with
	// the rest of the fleet — resident, warm or single-owner burst — and this
	// package has no basis on which to decide it. Guessing one would silently
	// place a large model in a tier the operator did not choose.
	Class vrambroker.Class
	// ResidentMemoryBytes is what the streaming path holds in memory while the
	// remainder streams from disk. This is the admission figure: asking for the
	// model's full in-memory requirement would refuse the very hosts this path
	// exists to serve, and asking for the disk footprint would ask for memory
	// the model never needs.
	ResidentMemoryBytes uint64
	// StorageBytes is the full on-disk footprint, carried so a caller can state
	// what the model occupies. It is not an admission figure — the broker gates
	// memory — but it is the axis this path trades into and it is not derivable
	// from the memory figure (D2).
	StorageBytes uint64
	// Tradeoff is what this path costs, carried from the choice. A session that
	// does not know it is slow by construction cannot tell anyone downstream
	// that the slowness is the deal it took, not a fault.
	Tradeoff *Tradeoff
}

// PlanLaunch derives the plan for starting choice's model under class.
//
// It refuses a choice that is not the streaming path, and refuses a choice and
// entry that are not the same model. Both are programming errors rather than
// conditions of the host, so they are plain errors and not refusals: a
// *Refusal says something about a user's machine, and neither of these does.
func (c Chooser) PlanLaunch(choice Choice, e catalogue.Entry, class vrambroker.Class) (LaunchPlan, error) {
	if choice.Runtime != catalogue.RuntimeStreaming {
		return LaunchPlan{}, fmt.Errorf("%w: choice runtime is %q", ErrNotStreamingChoice, choice.Runtime)
	}
	if choice.ModelID != e.ModelID || choice.Variant != e.Variant {
		return LaunchPlan{}, fmt.Errorf("%w: choice is %q, entry is %q",
			ErrChoiceEntryMismatch, choiceIdentity(choice), e.Identity())
	}
	return LaunchPlan{
		Identity:            e.Identity(),
		Class:               class,
		ResidentMemoryBytes: c.Streaming.ResidentMemoryBytes(e),
		StorageBytes:        c.Streaming.StorageBytes(e),
		Tradeoff:            choice.Tradeoff,
	}, nil
}

// choiceIdentity renders the model a choice was made for, for the mismatch
// error above. It is the same model:variant shape the catalogue entry uses, so
// the two sides of that error are directly comparable by eye.
func choiceIdentity(c Choice) string {
	if c.Variant == "" {
		return c.ModelID
	}
	return c.ModelID + ":" + c.Variant
}

// Launcher runs the lifecycle. Its components are injected so the ordering and
// the release paths are testable without a card and without the streaming
// runtime itself.
type Launcher struct {
	// Admit is the admission gate. Required — a launcher without one would
	// start processes against no budget at all.
	Admit Admitter
	// Health decides when a started process is serving. Required — without one
	// the launcher would report anything it managed to start as ready.
	Health HealthProbe
	// HealthBudget is how long a start is given to answer. Streamed weights load
	// slowly by construction, so this is generous by nature; what it must not be
	// is unbounded, or a process that will never answer is never cleaned up.
	HealthBudget time.Duration
	// HealthInterval is the gap between probes. Zero means the default.
	HealthInterval time.Duration
}

const (
	defaultHealthBudget   = 10 * time.Minute
	defaultHealthInterval = 2 * time.Second
)

func (l Launcher) budget() time.Duration {
	if l.HealthBudget <= 0 {
		return defaultHealthBudget
	}
	return l.HealthBudget
}

func (l Launcher) interval() time.Duration {
	if l.HealthInterval <= 0 {
		return defaultHealthInterval
	}
	return l.HealthInterval
}

// Session is a running streaming model and the reservation it holds. Close
// returns both, in that order.
type Session struct {
	Identity string
	Class    vrambroker.Class
	// Tradeoff is what this session's path costs, carried from the plan.
	Tradeoff *Tradeoff

	process Process
	lease   Lease
	once    sync.Once
	err     error
}

// Close stops the process and then releases the lease.
//
// The order is deliberate: the reservation covers a process that is still using
// the capacity until it is gone, so releasing first would advertise capacity
// that is still occupied and could admit a second workload into it.
//
// A failed stop is REPORTED but does not withhold the release. The caller has
// an orphan process to deal with either way; keeping the reservation as well
// would punish every other workload on the host for one process's bad exit.
//
// Close is idempotent. It is reached from a defer, from a signal handler and
// from an error path, sometimes all three, and returning capacity twice that
// was taken once is worse than not returning it at all.
func (s *Session) Close(ctx context.Context) error {
	s.once.Do(func() {
		s.err = stopThenRelease(ctx, s.process, s.lease)
	})
	return s.err
}

// stopThenRelease is the one teardown, used by both the failure paths inside
// Launch and by Close. Having a single implementation is what guarantees a
// launch that fails halfway cleans up the same way a healthy session does —
// two teardowns would eventually differ, and the one that ran less often would
// be the one that leaked.
func stopThenRelease(ctx context.Context, p Process, lease Lease) error {
	var err error
	if p != nil {
		err = p.Stop(ctx)
	}
	if lease != nil {
		lease.Release()
	}
	return err
}

// Launch admits, starts, and waits for health, returning a Session that holds
// the reservation for as long as the model serves.
//
// Every failure after admission tears down before returning, so no path out of
// this function leaves a lease held or a process running. The error returned is
// the one that caused the failure, not the one from the cleanup: a caller
// needs to know the runtime never became healthy, not that stopping it
// afterwards also had trouble.
func (l Launcher) Launch(ctx context.Context, plan LaunchPlan, p Process) (*Session, error) {
	if l.Admit == nil || l.Health == nil || p == nil {
		return nil, ErrIncompletePlan
	}

	// (1) Admission first. Nothing is started against a budget that has not
	// granted room for it.
	lease, err := l.Admit.Acquire(ctx, plan.Class, int64(plan.ResidentMemoryBytes))
	if err != nil {
		// The gate's own error travels out intact — over-budget, burst slot
		// taken, budget unreadable and thermally unsafe call for different
		// responses, and a replaced error takes that choice away from the caller.
		return nil, err
	}

	// (2) Start. From here every exit tears down.
	if err := p.Start(ctx); err != nil {
		// Stop as well as release: a start that reported failure may still have
		// left something behind, and the teardown is cheap compared with an
		// orphan holding the card.
		_ = stopThenRelease(ctx, p, lease)
		return nil, fmt.Errorf("runtime: starting %s: %w", plan.Identity, err)
	}

	// (3) Health, within a bounded budget.
	if err := l.awaitHealthy(ctx); err != nil {
		_ = stopThenRelease(ctx, p, lease)
		return nil, err
	}

	return &Session{
		Identity: plan.Identity,
		Class:    plan.Class,
		Tradeoff: plan.Tradeoff,
		process:  p,
		lease:    lease,
	}, nil
}

// awaitHealthy polls until the process answers, the budget expires, or the
// caller's context ends.
//
// It probes once before waiting: a runtime that is already up should not be
// made to wait out an interval for the launcher's benefit.
func (l Launcher) awaitHealthy(ctx context.Context) error {
	deadline, cancel := context.WithTimeout(ctx, l.budget())
	defer cancel()

	ticker := time.NewTicker(l.interval())
	defer ticker.Stop()

	for {
		if l.Health.Healthy(deadline) {
			return nil
		}
		select {
		case <-deadline.Done():
			// The caller's own cancellation is reported as itself; only an
			// expired budget is ErrNotHealthy, because only that is a statement
			// about the runtime rather than about the caller.
			if ctx.Err() != nil {
				return fmt.Errorf("runtime: waiting for health: %w", ctx.Err())
			}
			return fmt.Errorf("%w: %s", ErrNotHealthy, l.budget())
		case <-ticker.C:
		}
	}
}
