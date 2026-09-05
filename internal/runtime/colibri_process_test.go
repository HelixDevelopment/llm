package runtime_test

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/HelixDevelopment/HelixLLM/internal/runtime"
	"github.com/HelixDevelopment/HelixLLM/internal/vrambroker"
	"github.com/stretchr/testify/require"
)

// The T070 launch seam, wired. These exercise the REAL wiring: a real
// operating-system process is launched through the real ExecProcess, readiness
// is read from the process's real stderr (the R1-verified contract: the
// verbatim "OpenAI-compatible API listening on http://{host}:{port}/v1" line,
// NOT an HTTP probe — the socket binds before the engine finishes loading), and
// the launcher's lease discipline is asserted end to end.
//
// What they do NOT do is run Colibri itself, for the same reason
// colibri_test.go states: there is no Colibri build on this host and fetching
// and building one is not something a unit test may do. The launcher binary is
// configuration precisely so this seam is exercisable with a script that
// honours the same contract, and the gap — no run against a real Colibri
// build — is stated in the report rather than papered over.

// Compile-time conformance: the T070 wiring satisfies the seams the lifecycle
// was built around. If either side drifts, this file fails to build.
var _ runtime.Process = (*runtime.ColibriProcess)(nil)
var _ runtime.HealthProbe = (*runtime.ColibriHealth)(nil)

// freePort reserves an ephemeral port and releases it, so the fake launcher can
// bind nothing yet still print the address the readiness contract names. The
// port number only has to be honest in the stderr line the health probe reads.
func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer func() { _ = ln.Close() }()
	return ln.Addr().(*net.TCPAddr).Port
}

// fakeColibriLauncher writes a shell script that honours the Colibri launch
// contract's readiness half: it prints the verbatim listening line and stays
// alive, like a serving engine. The sleep is exec'd so the script process IS
// the sleeper — Stop's signal lands on the real process, and no orphaned
// grandchild is left holding the readiness pipe after teardown.
func fakeColibriLauncher(t *testing.T, port int) string {
	t.Helper()
	sh := shell(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "coli")
	script := fmt.Sprintf("#!%s\necho \"OpenAI-compatible API listening on http://127.0.0.1:%d/v1\" >&2\nexec sleep 3600\n", sh, port)
	require.NoError(t, os.WriteFile(path, []byte(script), 0o755))
	return path
}

// TestLauncherRefusesWithoutBinary.
//
// The honest refusal is the whole point of the config-presence gate: a launcher
// with no binary configured must say so, naming the missing piece and the
// remedy, and must never stand up a fake process to keep the lifecycle quiet.
// The refusal still releases the lease it admitted — the release-on-every-exit
// invariant holds on the way out too.
func TestLauncherRefusesWithoutBinary(t *testing.T) {
	rec := &recorder{}
	admitter := &fakeAdmitter{rec: rec}

	choice, entry := streamingChoice(t)
	plan, err := runtime.NewChooser().PlanLaunch(choice, entry, vrambroker.ClassCoder)
	require.NoError(t, err)

	proc := &runtime.ColibriProcess{
		ModelDir:  t.TempDir(),
		Host:      "127.0.0.1",
		Port:      freePort(t),
		StopGrace: time.Second,
	}

	l := launcher(admitter, &runtime.ColibriHealth{Proc: proc}, 10*time.Second)
	session, err := l.Launch(context.Background(), plan, proc)

	require.Nil(t, session, "a refused launch must not hand back a session")
	require.ErrorIs(t, err, runtime.ErrColibriBinaryNotConfigured,
		"the refusal is a closed-set error, not a wrapped start failure")
	require.ErrorContains(t, err, "HELIX_COLIBRI_BIN",
		"the refusal names the missing binary and the remedy")
	require.Equal(t, 1, admitter.lease.releases,
		"the lease taken before the refusal is released on the way out")
	require.Contains(t, rec.events, "lease.release")
}

// TestLauncherLaunchesWithBinary.
//
// The full lifecycle on the real wiring: a real subprocess is started through
// ExecProcess, readiness is recognised from its real stderr (the listening
// line, not an HTTP probe), the session is handed back holding its lease, and
// Close tears the process down and releases. The process must be genuinely
// gone afterwards — a Stop that only pretended would leave the fake launcher
// sleeping for an hour.
func TestLauncherLaunchesWithBinary(t *testing.T) {
	port := freePort(t)
	rec := &recorder{}
	admitter := &fakeAdmitter{rec: rec}

	choice, entry := streamingChoice(t)
	plan, err := runtime.NewChooser().PlanLaunch(choice, entry, vrambroker.ClassCoder)
	require.NoError(t, err)

	proc, err := runtime.NewColibriProcess(runtime.ColibriConfig{
		Binary:    fakeColibriLauncher(t, port),
		ModelDir:  t.TempDir(),
		Host:      "127.0.0.1",
		Port:      port,
		StopGrace: 2 * time.Second,
	})
	require.NoError(t, err)

	l := launcher(admitter, &runtime.ColibriHealth{Proc: proc}, 30*time.Second)
	session, err := l.Launch(context.Background(), plan, proc)

	require.NoError(t, err, "a launcher honouring the stderr contract must be reported ready")
	require.NotNil(t, session)
	require.True(t, proc.Ready(), "readiness came from the stderr listening line")
	require.Zero(t, admitter.lease.releases, "a healthy session still holds its reservation")

	require.NoError(t, session.Close(context.Background()))
	require.Equal(t, 1, admitter.lease.releases, "closing returns the reservation")

	require.Eventually(t, func() bool { return proc.Dead() },
		10*time.Second, 10*time.Millisecond,
		"the stopped process must actually be gone, not merely asked to leave")
}

// TestColibriHealthTreatsEngineDeathAsTerminal.
//
// R1 contract item 4: the engine child dying mid-serve is TERMINAL. When the
// process exits and its stderr reaches EOF, the health probe must report not
// ready and the process must report itself dead — never a stale "ready" for a
// process that no longer exists.
func TestColibriHealthTreatsEngineDeathAsTerminal(t *testing.T) {
	port := freePort(t)
	sh := shell(t)
	dir := t.TempDir()
	binary := filepath.Join(dir, "coli")
	// Honours the contract, then dies: one listening line, then exit. A process
	// manager reading Dead() sees the terminal state instead of polling a corpse.
	script := fmt.Sprintf("#!%s\necho \"OpenAI-compatible API listening on http://127.0.0.1:%d/v1\" >&2\nexit 1\n", sh, port)
	require.NoError(t, os.WriteFile(binary, []byte(script), 0o755))

	proc, err := runtime.NewColibriProcess(runtime.ColibriConfig{
		Binary:    binary,
		ModelDir:  t.TempDir(),
		Host:      "127.0.0.1",
		Port:      port,
		StopGrace: time.Second,
	})
	require.NoError(t, err)

	health := &runtime.ColibriHealth{Proc: proc}
	require.NoError(t, proc.Start(context.Background()))

	require.Eventually(t, func() bool { return proc.Dead() },
		10*time.Second, 10*time.Millisecond,
		"stderr EOF after the engine exits must surface as dead")
	require.False(t, health.Healthy(context.Background()),
		"a dead engine is never healthy, even though it once printed the listening line")
	require.Error(t, proc.ExitErr(),
		"the unexpected exit is reported, so a process manager can log why it died")
}
