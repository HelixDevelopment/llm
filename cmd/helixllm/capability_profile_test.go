package main

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/HelixDevelopment/HelixLLM/internal/capability"
)

// F5 finding: the measured capability profile was only visible as a handful of
// boot log lines, with no way to dump the full HostCapabilityProfile. This
// pins the JSON seam: every measured field a caller needs to reason about
// selection must round-trip through the printer, and the output must be real
// JSON (parseable into the profile struct), not prose.

func TestPrintCapabilityProfileEmitsMeasuredFields(t *testing.T) {
	var buf bytes.Buffer
	// Measurement on a GPU-less CI host is legitimately partial; the printer
	// still emits whatever was genuinely measured, so this asserts the seam,
	// not this host's hardware.
	_ = printCapabilityProfile(context.Background(), &buf)

	out := buf.Bytes()
	if len(out) == 0 {
		t.Fatal("the capability seam must emit the profile, not nothing")
	}

	var p capability.HostCapabilityProfile
	if err := json.Unmarshal(out, &p); err != nil {
		t.Fatalf("the emitted profile must be real JSON: %v\nbody: %.200s", err, out)
	}

	for _, field := range []string{
		`"HostIdentity"`,
		`"CPU"`,
		`"MemoryTotal"`,
		`"MemoryAvailable"`,
		`"AcceleratorState"`,
		`"Accelerators"`,
		`"StorageAvailable"`,
		`"MeasuredAt"`,
		`"MeasurementComplete"`,
	} {
		if !bytes.Contains(out, []byte(field)) {
			t.Errorf("capability JSON is missing measured field %s", field)
		}
	}
}
