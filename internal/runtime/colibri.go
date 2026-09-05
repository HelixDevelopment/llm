package runtime

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"
)

// The disk-streaming runtime, as an adopted external process.
//
// WHAT IT IS. The streaming runtime is Colibri (https://github.com/JustVugg/colibri,
// Apache-2.0): a specialised inference engine that keeps a mixture-of-experts
// model's dense backbone resident in RAM and streams the routed-expert weights
// from local disk on demand. That is what lets a 744B-parameter model run on a
// 16 GiB host, and it is also why it is slow — the throughput cost is not a
// tuning problem, it is the mechanism.
//
// WHY THERE IS NO IMPORT. Colibri is pure C with no Go binding. Adopting it is
// a launch-and-lifecycle integration, not a library dependency, so the module's
// go.mod is unchanged by its adoption. Its build is `git clone` + `make -C
// colibri/c` with GCC or Clang and OpenMP, and it has no runtime dependencies
// of its own.
//
// WHAT IS CONFIGURED RATHER THAN COMPILED IN. The binary's path, its arguments,
// and the address it answers health on are supplied by the deployment. They are
// not constants here for two reasons. First, they are genuinely not knowable
// from what has been established about the runtime: the research that authorised
// this adoption records the repository, the licence, the build command, the
// supported-family roster and the per-family RAM and disk figures — it does not
// record a serving invocation or a health endpoint, and inventing one would be
// stating something about the runtime that nobody has checked. Second, a host
// path or address compiled into a source file is the hardcoded-distribution-host
// pattern the project forbids outright.
//
// So this file supplies the two REAL implementations the lifecycle needs — an
// operating-system process and an HTTP health check — and the deployment names
// what to run. Neither is a stand-in: ExecProcess starts and stops a real
// process, HTTPHealth makes a real request.

// ExecProcess runs the streaming runtime as an operating-system process.
//
// It deliberately does NOT use exec.CommandContext. That would bind the
// process's lifetime to the context passed to Start — the LAUNCH context, which
// is scoped to bringing the model up and is cancelled as soon as it has. The
// process must outlive its own launch and end only when Stop says so, which is
// what keeps the single teardown path in Session.Close the only way it dies.
type ExecProcess struct {
	// Binary is the path to the built runtime executable.
	Binary string
	// Args are its arguments — the model container to serve, the address to
	// listen on, and whatever else the deployment's build of it takes.
	Args []string
	// Dir is the working directory, empty for the caller's own.
	Dir string
	// Env is the environment, nil to inherit the caller's.
	Env []string
	// Stdout and Stderr receive the runtime's output. A streaming runtime that
	// is loading reports its progress there, and discarding it turns the most
	// common failure — a load that never finishes — into a silent timeout with
	// nothing to read afterwards.
	Stdout, Stderr *os.File
	// StopGrace is how long a signalled process is given to exit before it is
	// killed. Zero means the default. It is generous by nature: a streaming
	// runtime may be mid-read of a very large file, and killing it immediately
	// risks leaving that file's cache state behind.
	StopGrace time.Duration

	mu      sync.Mutex
	cmd     *exec.Cmd
	done    chan error
	waitErr error
	waitSet bool
}

const defaultStopGrace = 30 * time.Second

// Errors reported by ExecProcess.
var (
	// ErrNoBinary: nothing was named to run. Refused rather than defaulted —
	// there is no binary this package could pick on the caller's behalf without
	// guessing where their build of the runtime lives.
	ErrNoBinary = errors.New("runtime: no streaming-runtime binary named to launch")
	// ErrAlreadyStarted: Start was called twice on one process value. The second
	// call would abandon the first process, which would then be running with
	// nothing holding a handle to stop it.
	ErrAlreadyStarted = errors.New("runtime: streaming-runtime process is already started")
)

// Start launches the process. It returns once the process is RUNNING, which is
// not the same as ready — a streaming runtime is running long before it can
// answer, and readiness is the health probe's question.
//
// The context is honoured for the start itself and then deliberately let go of;
// see the type's doc comment.
func (p *ExecProcess) Start(ctx context.Context) error {
	if p.Binary == "" {
		return ErrNoBinary
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cmd != nil {
		return ErrAlreadyStarted
	}

	cmd := exec.Command(p.Binary, p.Args...)
	cmd.Dir = p.Dir
	cmd.Env = p.Env
	if p.Stdout != nil {
		cmd.Stdout = p.Stdout
	}
	if p.Stderr != nil {
		cmd.Stderr = p.Stderr
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("runtime: starting %s: %w", p.Binary, err)
	}

	// Wait is called exactly once, here, in its own goroutine. Its result is
	// kept so Stop can report how the process actually ended rather than
	// assuming it ended well, and so the process is reaped instead of left a
	// zombie if Stop is never reached. It is also recorded under the mutex
	// (waitErr/waitSet) so WaitResult can report the same ending without
	// draining the channel Stop reads.
	p.cmd = cmd
	p.done = make(chan error, 1)
	p.waitErr, p.waitSet = nil, false
	go func(c *exec.Cmd, done chan<- error) {
		err := c.Wait()
		p.mu.Lock()
		p.waitErr, p.waitSet = err, true
		p.mu.Unlock()
		done <- err
	}(cmd, p.done)

	return nil
}

// WaitResult reports how the process ended once Wait has returned, without
// consuming the result Stop reads from the done channel. A caller that needs
// the exit status independently of the teardown — a stderr watch reporting
// engine death, say — uses this, because two readers on one channel would
// race and the loser would hang. The second return is false until the process
// has actually exited.
func (p *ExecProcess) WaitResult() (error, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.waitErr, p.waitSet
}

// Stop ends the process: signal, wait out the grace period, then kill.
//
// A process that has already exited is not an error — Stop is reached from the
// teardown of a launch that failed because the runtime died on its own, and
// reporting "it is not running" there would replace the real diagnosis with a
// complaint about the cleanup.
//
// Stop is safe to call on a process that was never started, for the same reason
// Session.Close is: the teardown runs on failure paths that may not have got as
// far as starting anything.
func (p *ExecProcess) Stop(ctx context.Context) error {
	p.mu.Lock()
	cmd, done := p.cmd, p.done
	p.cmd, p.done = nil, nil
	p.mu.Unlock()

	if cmd == nil || cmd.Process == nil {
		return nil
	}

	// Ask first. A runtime mid-read of a multi-hundred-gigabyte file is given
	// the chance to close it.
	//
	// os.Interrupt is the portable request; on Windows it is not deliverable and
	// the kill below is what actually ends the process. That is a real platform
	// difference and is left visible rather than papered over.
	_ = cmd.Process.Signal(os.Interrupt)

	grace := p.StopGrace
	if grace <= 0 {
		grace = defaultStopGrace
	}
	timer := time.NewTimer(grace)
	defer timer.Stop()

	select {
	case err := <-done:
		return exitError(err)
	case <-timer.C:
	case <-ctx.Done():
	}

	// It did not go. Kill, then still wait for it, so the function does not
	// return while the process is in the middle of dying and the capacity its
	// lease is about to release is still occupied.
	if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		return fmt.Errorf("runtime: killing %s: %w", p.Binary, err)
	}
	return exitError(<-done)
}

// exitError reports how the process ended, discarding the two endings that are
// not faults: a clean exit, and an exit caused by the signal Stop just sent.
func exitError(err error) error {
	if err == nil {
		return nil
	}
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		// The process was ended by this teardown. Reporting the signal it was
		// asked to die from as a failure would make every ordinary shutdown
		// look like one.
		return nil
	}
	return err
}

// HTTPHealth reports whether the streaming runtime is answering on an HTTP
// endpoint.
//
// Unhealthy covers every way of not answering — connection refused while it is
// still binding, a non-2xx while it is still loading weights, a body that never
// arrives. They are one answer here on purpose: the caller polls this, and for
// most of a streaming load the true answer is "not yet". Distinguishing kinds of
// not-ready would invite treating one of them as fatal, and a slow load would
// then be reported as a broken runtime.
type HTTPHealth struct {
	// URL is the endpoint to probe.
	URL string
	// Client is the HTTP client. Nil means a client with PerProbeTimeout.
	Client *http.Client
	// PerProbeTimeout bounds a single probe so one hung connection cannot
	// consume the whole health budget in a single attempt. Zero means the
	// default.
	PerProbeTimeout time.Duration
}

const defaultPerProbeTimeout = 5 * time.Second

// Healthy makes one real request and reports whether it answered 2xx.
func (h HTTPHealth) Healthy(ctx context.Context) bool {
	if h.URL == "" {
		return false
	}

	timeout := h.PerProbeTimeout
	if timeout <= 0 {
		timeout = defaultPerProbeTimeout
	}
	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, h.URL, nil)
	if err != nil {
		return false
	}

	client := h.Client
	if client == nil {
		client = &http.Client{Timeout: timeout}
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()

	return resp.StatusCode >= 200 && resp.StatusCode < 300
}
