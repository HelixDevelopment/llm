package main

// Measured model selection for the VIDEO-GENERATION lane (T078 / FR-056).
//
// The rule this file exists to enforce: WHICH model runs is decided from the
// MEASURED host, never from a static preconfigured value. Configuration may
// still say WHERE the catalogue lives (HELIXLLM_CATALOGUE_DIR), what the user
// has DECLARED about their usage (HELIXLLM_DECLARED_USAGE) and which options
// they FORBID (VIDEOGEN_FORBID_MODELS). None of those names the model.
//
// Video has NO processor-viable option — every recorded entry sets
// requires_accelerator, because diffusion video generation without an
// accelerator is not slow-but-usable, it is unusable. A host with no
// accelerator is therefore told what it LACKS and offered nothing. That
// refusal is the correct answer, not a gap: an offer that would start and then
// be unusable is the failure being avoided.
//
// The DECISION ITSELF now lives in internal/laneboot, shared by all four boot
// lanes. It was previously carried here as one of four byte-identical copies —
// the duplication notice this comment used to hold. What remains below is the
// half that is genuinely the video lane's: where its weights come from, which
// diffusion pipeline and precision it serves, the clip shape an option was
// selected at, and its own not-servable exit code.
//
// The properties laneboot enforces — no fixed default, and a pin that
// constrains rather than bypasses — are documented there and are not restated
// here, so there is one place to read them and one place to change them.

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/HelixDevelopment/HelixLLM/internal/capability"
	"github.com/HelixDevelopment/HelixLLM/internal/catalogue"
	"github.com/HelixDevelopment/HelixLLM/internal/laneboot"
	"github.com/HelixDevelopment/HelixLLM/internal/selection"
)

// exitNotServable is this lane's own refusal code: the host CAN serve a chosen
// option but this runtime cannot serve that BUILD of it. The codes the shared
// decision reaches (20-23) live in laneboot; this one is the video lane's,
// because only the video lane knows which diffusion builds it serves.
const exitNotServable = 24

// weightsDirKey is the operator override naming the host directory that will
// hold this lane's weights. It names WHERE, never WHICH.
const weightsDirKey = "VIDEOGEN_WEIGHTS_DIR"

// weightsVolume is the named compose volume this lane's weights actually land
// in: compose.videogen.yml mounts it at /models/hf with HF_HOME=/models/hf.
// The storage axis of the decision must be measured on THIS filesystem —
// the volume's mountpoint, or the runtime's storage root before the volume
// exists — never on the process working directory (EX-28: an empty WeightsDir
// there measured /dev/shm at 15325 MiB while the volume's filesystem held
// 1120821 MiB — a 73x different fit answer on one host).
const weightsVolume = "helixllm-videogen-cache"

// weightsDir resolves the filesystem the decision's storage axis is measured
// on. Every failure path refuses with the host-not-measured code: the decision
// may not proceed on a number that describes the wrong filesystem.
func weightsDir(ctx context.Context) (string, error) {
	dir, err := laneboot.ResolveWeightsDir(ctx, weightsDirKey, weightsVolume)
	if err != nil {
		return "", laneboot.ExitErr(laneboot.ExitHostNotMeasured, "CANNOT-CHOOSE: %v", err)
	}
	return dir, nil
}

// choice is one decided model, with the measurement it was decided from and the
// artefacts on this host that serve it.
type choice struct {
	Option  selection.Option
	Entry   catalogue.Entry
	Profile capability.HostCapabilityProfile
	// Usage is the purpose the terms were applied against, so a report can
	// state what the licence permits rather than merely naming it.
	Usage catalogue.UsagePurpose

	// Repository is where the chosen model's weights are obtained from, taken
	// from the chosen entry's recorded source. It is DERIVED from the decision,
	// never read from configuration.
	Repository string
	// Backend is the diffusion pipeline this runtime loads for the chosen
	// model (WanPipeline / LTXPipeline).
	Backend string
	// Precision is the serving precision the chosen build implies.
	Precision string
	// Shape is the clip the chosen option was selected AT — the frame size,
	// count and rate its recorded memory figure belongs to.
	Shape videoShape
}

// repositoryFor is where the chosen model's weights are obtained from.
//
// It reads Entry.Source and nothing else. Source is the field the FR-012 /
// SC-011 acquisition allowlist is checked against, so deriving the repository
// from anywhere else would hand the runtime a weight location that no gate ever
// saw. In particular `annotations.weights_repo` is NOT consulted: annotations
// are carried and shown, never read to make a decision (catalogue.Entry's own
// contract), and they are not validated.
//
// KNOWN DATA GAP, stated rather than worked around (§11.4.6): no entry in
// internal/catalogue/data/video.yaml records `source:` today — that file says so
// deliberately, because `source` is what the acquisition gate checks and
// choosing its value is a data decision. Until an entry records one, this
// function refuses it and the lane boots nothing. That is the honest outcome:
// the alternative is pulling a repository nobody validated.
func repositoryFor(e catalogue.Entry) (string, error) {
	src := strings.TrimSpace(e.Source)
	if src == "" {
		return "", fmt.Errorf("catalogue entry %s records no weight source, so there is no validated "+
			"repository to serve (annotations are not read for this: they are unvalidated). "+
			"Remedy: record `source:` on the entry", e.Identity())
	}
	const host = "https://huggingface.co/"
	if !strings.HasPrefix(src, host) {
		return "", fmt.Errorf("weight source %q for %s is not a model-repository reference this runtime can pull",
			src, e.Identity())
	}
	repo := strings.Trim(strings.TrimPrefix(src, host), "/")
	if strings.Count(repo, "/") != 1 || repo == "" {
		return "", fmt.Errorf("weight source %q for %s is not an <owner>/<model> repository", src, e.Identity())
	}
	return repo, nil
}

// servingBackends maps a recorded model to the diffusion pipeline this runtime
// implements for it (services/videogen/videogen_server.py: WanPipeline or
// LTXPipeline). It is the runtime declaring what it can serve, in the same
// shape as servingPrecisions below.
//
// It names models, but it can only ever REMOVE a candidate the measurement
// offered — never introduce one, and never supply weights. A model absent from
// this map is reported as not-servable-here rather than sent to a pipeline that
// cannot load it. Which model runs is still decided by the measurement; this
// only answers "and can this runtime serve that one at all".
//
// Keyed by model id because that is what distinguishes the two pipelines:
// `architecture` cannot, both WAN and LTX are recorded as diffusion.
var servingBackends = map[string]string{
	"wan2.2-ti2v-5b":  "wan",
	"wan2.2-t2v-a14b": "wan",
	"ltx-video-13b":   "ltx",
}

// backendFor reports which pipeline this runtime would load for the chosen
// model.
func backendFor(o selection.Option) (string, error) {
	b, ok := servingBackends[strings.ToLower(strings.TrimSpace(o.ModelID))]
	if !ok {
		return "", fmt.Errorf("this runtime implements no diffusion pipeline for %s "+
			"(it serves: %s)", o.ModelID, strings.Join(sortedKeys(servingBackends), ", "))
	}
	return b, nil
}

// servingPrecisions are the weight formats the video runtime actually
// implements (services/videogen/videogen_server.py). A build quantised outside
// this set is not servable HERE — reported honestly rather than silently
// coerced into a precision the entry's memory figure was never measured at.
var servingPrecisions = map[string]string{
	"fp8":  "fp8",
	"bf16": "bf16",

	// "gguf-q4" is DELIBERATELY ABSENT, and removing it was a regression fix.
	//
	// services/videogen/videogen_server.py refuses gguf-q4 in
	// _UNIMPLEMENTED_PRECISIONS: the service has no GGUF load path at all — it
	// builds pipelines only through from_pretrained, which resolves a
	// diffusers-format repository and cannot read single-file .gguf weights.
	//
	// This list and that one are two independent statements about the same
	// question, and they disagreed. Nothing noticed while most-capable-first
	// ordering happened to rank a servable fp8 build first. When ordering
	// changed to cheapest-first, the cheaper gguf-q4 entry moved to the front
	// and the lane began choosing a build the runtime refuses: admit the VRAM,
	// compose up, 503 for the full health timeout, exit 4, holding the lease
	// throughout.
	//
	// The service is authoritative here — it is the thing that actually loads
	// weights. If a GGUF load path is added there, add the precision back HERE
	// in the same change, and see the agreement test that now pins the two
	// lists together.
}

// precisionFor reports the serving precision the chosen build implies.
//
// It reads the recorded Descriptor.Quantisation rather than splitting the
// variant string: the variant is an identity ("fp8-480p"), the quantisation is
// the validated field that says what the weights actually are.
func precisionFor(o selection.Option) (string, error) {
	q := strings.ToLower(strings.TrimSpace(o.Descriptor.Quantisation))
	if q == "" {
		return "", fmt.Errorf("build %q of %s records no quantisation, so the serving precision is unknown",
			o.Variant, o.ModelID)
	}
	p, ok := servingPrecisions[q]
	if !ok {
		return "", fmt.Errorf("quantisation %q of %s is not one this runtime serves (%s)",
			q, o.ModelID, strings.Join(sortedKeys(servingPrecisions), ", "))
	}
	return p, nil
}

// videoShape is the clip the chosen entry is offered to produce: the frame
// size, count and rate the option was selected AT. Serving some other shape
// would be serving a configuration whose memory figure nobody measured.
type videoShape struct {
	Size      string // WxH, as the runtime expects it
	NumFrames int
	FPS       int
}

// videoShapeFor reads the shape from the chosen option's recorded video output.
//
// A missing or fractional rate is refused rather than rounded: the runtime
// parses the rate as an integer, so a fractional one would abort the container
// at start-up, and rounding it would serve a clip nobody recorded.
func videoShapeFor(o selection.Option) (videoShape, error) {
	v := o.Expected.Video
	if !v.Recorded() {
		return videoShape{}, fmt.Errorf("%s records no video output, so there is no clip shape to serve",
			o.Identity)
	}
	fps := int(v.FramesPerSecond)
	if float64(fps) != v.FramesPerSecond {
		return videoShape{}, fmt.Errorf("%s records a fractional rate (%g fps) this runtime cannot serve",
			o.Identity, v.FramesPerSecond)
	}
	return videoShape{
		Size:      fmt.Sprintf("%dx%d", v.FrameWidth, v.FrameHeight),
		NumFrames: v.FrameCount,
		FPS:       fps,
	}, nil
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
