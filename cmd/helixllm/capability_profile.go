package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/HelixDevelopment/HelixLLM/internal/capability"
)

// The F5 finding: the measured host capability profile — the single input
// selection reasons from — was only visible as a handful of boot log lines.
// This is the dump seam: measure the host and print the full
// HostCapabilityProfile as JSON so an operator (or an agent acting for one)
// can see exactly what the chooser sees.

// printCapabilityProfile measures this host and writes the measured
// HostCapabilityProfile as indented JSON to w. The profile is emitted even
// when measurement was partial — an honestly incomplete reading with
// MeasurementComplete=false is more useful than no reading at all — but the
// measurement error is then returned alongside it, so the caller can decide
// whether the gap matters for what they are diagnosing.
func printCapabilityProfile(ctx context.Context, w io.Writer) error {
	p, err := capability.Measure(ctx, capability.Options{})
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if encErr := enc.Encode(p); encErr != nil {
		return fmt.Errorf("encoding capability profile: %w", encErr)
	}
	if err != nil {
		return fmt.Errorf("capability: measurement incomplete: %w", err)
	}
	return nil
}
