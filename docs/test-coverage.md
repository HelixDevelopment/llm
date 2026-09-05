# HelixLLM Test-Coverage Ledger

**Round 215 deliverable. Format and intent: CONST-050(B) coverage ledger
mandated by the constitution submodule §11.4.27 — every supported test
type accounted for, with the current runtime evidence pointer for each
row. CONST-048 release-gate sweep regenerates this document; manual
edits are permitted between sweeps to record fresh evidence.**

> Verbatim 2026-05-19 operator mandate (CONST-049 §11.4.17 anchor):
> *"all existing tests and Challenges do work in anti-bluff manner -
> they MUST confirm that all tested codebase really works as expected!
> We had been in position that all tests do execute with success and
> all Challenges as well, but in reality the most of the features does
> not work and can't be used! This MUST NOT be the case and execution
> of tests and Challenges MUST guarantee the quality, the completition
> and full usability by end users of the product!"*

---

## 1. Purpose and audience

This document is the **authoritative ledger** for HelixLLM's test posture.
A reader (operator, reviewer, CI-replacement script) should be able to
answer two questions in under sixty seconds:

1. **Does test type X exist for HelixLLM today?** — column "Status".
2. **What is the captured runtime evidence?** — column "Evidence" plus
   the per-row notes section below.

Rows are anchored to the constitution submodule's §11.4.27 enumeration so
the universal `verify-all-constitution-rules.sh` sweep (CONST-055) can
diff the expected row set against this document and flag silent omissions.

A row that says "PASS" without an Evidence pointer is a **CONST-035 /
Article XI §11.9 violation** of the same severity as a green CI badge
hiding a broken feature. Reviewers MUST refuse such rows.

---

## 2. Coverage matrix

The matrix below covers HelixLLM's seven primary surfaces against the
fourteen test types §11.4.27 enumerates. The "Surface" axis is the
narrowest meaningful slicing of HelixLLM's runtime — finer-grained
slicing (per-provider, per-mode) lives in the per-bank YAML files under
`challenges/banks/`.

| Surface \\ Test type           | Unit | Integration | E2E | Full-automation | Security | DDoS | Scaling | Chaos | Stress | Performance | Bench | UI | UX | Challenges | helix_qa |
|--------------------------------|:----:|:-----------:|:---:|:---------------:|:--------:|:----:|:-------:|:-----:|:------:|:-----------:|:-----:|:--:|:--:|:----------:|:--------:|
| Gateway (OpenAI/Anthropic API) | PASS | PASS        | PASS | PASS            | PASS     | PASS | PASS    | PASS  | PASS   | PASS        | PASS  | n/a | n/a | PASS       | PENDING  |
| Brain (LLM routing + llama.cpp)| PASS | PASS        | PASS | PASS            | PASS     | n/a  | PASS    | PASS  | PASS   | PASS        | PASS  | n/a | n/a | PASS       | PENDING  |
| Multi-provider fallback chain  | PASS | PASS        | PASS | PASS            | PASS     | PASS | n/a     | PASS  | PASS   | PASS        | PASS  | n/a | n/a | PASS       | PENDING  |
| Knowledge (RAG pipeline)       | PASS | PASS        | PASS | PASS            | PASS     | n/a  | n/a     | n/a   | n/a    | PASS        | n/a   | n/a | n/a | PASS       | PENDING  |
| Agents (ReAct + tools)         | PASS | PASS        | PASS | PASS            | PASS     | n/a  | n/a     | n/a   | n/a    | n/a         | n/a   | n/a | n/a | PASS       | PENDING  |
| Control plane (cluster mgmt)   | PASS | PASS        | n/a | PASS            | PASS     | n/a  | PASS    | PASS  | n/a    | n/a         | n/a   | TUI* | TUI* | PASS       | PENDING  |
| CLI (`helixllm` binary)        | PASS | PASS        | PASS | PASS            | n/a      | n/a  | n/a     | n/a   | n/a    | n/a         | n/a   | n/a | n/a | PASS       | PENDING  |

Legend:
- **PASS** — test type exists, last execution produced positive runtime
  evidence pointed to by §3 below.
- **PENDING** — test type incorporated at the submodule layer
  (`helix_qa/` autonomous-session integration in flight per CONST-050(B));
  no evidence captured yet for this submodule's banks. Tracked as a
  PENDING_FORENSICS row per §11.4.15 — closure requires evidence, not
  status flip.
- **n/a** — test type not applicable to the surface (e.g., DDoS against
  a stateless model-routing module, or UI for non-TUI surfaces).
- **TUI\*** — covered by the terminal-UI monitor (`helixllm --monitor`);
  visual-regression captures pending.

---

## 3. Evidence pointers (per row)

The list below maps every PASS cell in §2 to the captured runtime
evidence that justifies the status. The intent is that a reader can
re-execute the listed command and reproduce the same outcome without
guessing what "PASS" means in this codebase.

### 3.1 Unit tests (mocks permitted per CONST-050(A))

- **Command**: `make test-unit` (root: `dependencies/HelixDevelopment/HelixLLM/`)
- **Scope**: every `*_test.go` under `internal/...` plus `cmd/helixllm/challenges_test.go`.
- **Coverage threshold**: 91% — enforced by `make coverage`.
- **Mock policy**: PERMITTED only in `*_test.go` files invoked without
  any integration build tag. The `fakeTranslator` in
  `cmd/helixllm/challenges_test.go` is the canonical example —
  satisfies `i18n.TranslatorAPI`, lives in the unit-test source, never
  imported from production code (audit: `grep -rn "fakeTranslator"
  cmd/ internal/ | grep -v _test.go` returns empty).
- **Anti-bluff guard**: every unit test that exercises a user-facing
  message MUST assert the message reaches the translator (key + lang)
  rather than appearing as a bare English literal — see
  `TestRunChallenges_FailedToLoadBanks_UsesTranslator`.
- `internal/runtime/colibri_process_test.go` — T070 colibri launch seam
  (W2a-2): compile-time Process/HealthProbe conformance, honest
  no-binary refusal with lease release, real-subprocess
  launch→stderr-readiness→teardown, terminal engine-death detection.
- `cmd/helixllm/capability_profile_test.go` — F5 capability JSON seam
  (W2a-2): the measured HostCapabilityProfile round-trips through the
  printer as parseable JSON carrying every measured field.

### 3.2 Integration tests (no mocks per CONST-050(A))

- **Command**: `make test-integration`
- **Scope**: `tests/integration/` exercises gateway → brain → fallback →
  knowledge wiring against real Gin server, real llama.cpp endpoint
  (started via `make dev` or `docker compose -f docker-compose.enterprise.yml`),
  real Redis/Postgres backing services.
- **Evidence**: integration runs surface real HTTP/3 response bodies,
  real Prometheus counter deltas, real SSE event streams captured to
  `tests/integration/captures/` (gitignored — captures regenerate per
  run, asserts read live).

### 3.3 End-to-end tests

- **Command**: `make test-e2e`
- **Scope**: `-tags=e2e ./tests/integration/...` — full user-flow
  exercise from `curl` through gateway through provider chain back to
  user, no internal seams stubbed.
- **Anti-bluff**: every E2E test asserts the response body contains a
  provider-specific signature (real model output structure) — never
  passes on "got 200 OK".

### 3.4 Full-automation pipeline

- **Command**: `make test-automation` — `test-unit` → `test-integration`
  → `test-challenges` chained in one invocation.
- **Scope**: every test type in §3.1-3.3 plus the §3.13 challenge bank
  sweep, in the order a release-gate audit would run them.

### 3.5 Security tests

- **Command**: `make test-security` + `make scan-quick` (govulncheck +
  gosec) + `make scan-fs` (Trivy filesystem scan).
- **Scope**: authn/authz boundaries on every gateway endpoint, JWT
  signature verification, API-key constant-time comparison, rate-limit
  evasion, prompt-injection at the agents tool boundary, dependency CVE
  scan.

### 3.6 DDoS tests

- **Command**: `make test-stress` against the
  `challenges/banks/benchmarks/` bank with the request-flood profile.
- **Evidence**: captured per-request latency histogram; SLO assert is
  p99 stays below 5s under 1000 RPS on the gateway health endpoint.

### 3.7 Scaling tests

- **Command**: `./challenges/scripts/scaling_horizontal_challenge.sh`
- **Scope**: spawns multiple `helixllm` processes in `gateway` /
  `brain` / `knowledge` modes, asserts load balancing distributes
  requests, measures throughput-per-node delta.

### 3.8 Chaos tests

- **Command**: `make test-chaos` + `./challenges/scripts/chaos_failure_injection_challenge.sh`
- **Scope**: kill brain process mid-request, partition control-plane
  SSH connection, fill knowledge-store disk, skew clock on one node.
- **Evidence**: every chaos test asserts the system **recovers** with
  captured timing — never passes on "request returned an error during
  chaos" alone.

### 3.9 Stress tests

- **Command**: `make test-stress` + `make test-stress-go`
- **Scope**: sustained load above advertised throughput tier for ≥10
  minutes; asserts no memory leak (RSS growth ceiling), no goroutine
  leak (final goroutine count within 1.1× baseline).

### 3.10 Performance tests

- **Command**: `make test-performance` (`-tags=performance -timeout=5m`).
- **SLO baselines**: stored in `tests/performance/baselines/` — drift
  detection refuses to mark PASS when p95 latency exceeds baseline by
  more than 10%.

### 3.11 Benchmarking

- **Command**: `make test-benchmark` (Challenge-driven) + `make
  test-benchmark-go` (Go micro-benchmarks).
- **Evidence**: historical p95 trend stored under
  `reports/benchmarks/` for week-over-week drift detection.

### 3.12 UI / UX (terminal)

- **Command**: `./bin/helixllm --monitor` (tview-driven cluster monitor)
  + `./challenges/scripts/ui_terminal_interaction_challenge.sh`.
- **Scope**: tab navigation, status-pane refresh cadence, force-refresh
  binding, graceful shutdown on `Ctrl+C`.
- **UX coverage**: `./challenges/scripts/ux_end_to_end_flow_challenge.sh`
  — operator clones the repo, runs `make dev`, issues a chat request,
  observes streaming completion. Whole flow timed and asserted under
  thresholds.

### 3.13 Challenges (vasic-digital/Challenges integration)

- **Command**: `make test-challenges` — sweeps every YAML bank under
  `challenges/banks/`.
- **Bank inventory**: `api/`, `benchmarks/`, `chaos/`, `e2e/`, `llm/`,
  `performance/`, `rag/`, `regression/`, `safety/`, `security/`,
  `stress/`, `workflows/`.
- **Cross-cutting scripts**: 12 shell scripts under
  `challenges/scripts/`. Notable additions:
  - `helixllm_cli_challenge.sh` (round 215) — anti-bluff sweep of the
    CLI surface plus i18n migration verification plus paired-mutation
    self-test.
  - `multi_provider_fallback_challenge.sh` — provider × fallback
    wiring verification, runs unit + build.
  - `chaos_failure_injection_challenge.sh`,
    `ddos_health_flood_challenge.sh`, `scaling_horizontal_challenge.sh`,
    `stress_sustained_load_challenge.sh` — runtime resilience proofs.

### 3.14 helix_qa (HelixDevelopment/HelixQA integration)

- **Status**: PENDING per CONST-050(B) — autonomous-session bank
  registration for HelixLLM scheduled with the next helix_qa pointer
  bump. Tracked as a §11.4.15 PENDING_FORENSICS row; flipping to PASS
  requires captured wire evidence from at least one full autonomous QA
  session, not metadata-only confirmation.

---

## 4. Anti-bluff guarantees (Article XI §11.9 anchor)

Every row in §2 that reads PASS carries one of three evidence shapes:

1. **Captured artefact**: log, recording, JSON dump, latency histogram
   stored under the path noted in §3.
2. **Reproducible invocation**: a `make` target or shell script that
   any reviewer can re-run and observe identical output structure
   (content varies — wire evidence does not).
3. **Self-validating assertion**: a test that fails noisily when its
   premise is violated, with the failure mode documented in the
   per-row notes above.

Metadata-only / configuration-only / absence-of-error / grep-based
PASS without one of the three shapes above is a CONST-035 violation,
to be reopened per CONST-058 with `By: AI, Reason:
captured-evidence-contradicts` within 24 hours of detection.

---

## 5. How this document stays in sync

- **Trigger**: every Round-N close-out that touches HelixLLM runs
  `verify-all-constitution-rules.sh` (CONST-055). The sweep parses §2
  rows and asserts every expected (Surface × Test-type) cell either has
  a documented PASS evidence pointer in §3 or an explicit `PENDING`/`n/a`
  rationale.
- **Drift detection**: a Surface added to HelixLLM without a
  corresponding row in §2 is a CONST-048 violation. A test type added
  to §11.4.27 without a corresponding column in §2 is a CONST-049
  cascade-incomplete violation.
- **Reopens**: any row demoted from PASS to a non-terminal status
  (e.g., test broke under a refactor) MUST be reopened in
  `docs/issues/Issues.md` with CONST-058 `Reopened-Details:` block
  (`By`, `On`, `Reason`, `Evidence`).

---

## 6. References

- Constitution submodule §11.4.27 (CONST-050) — no-fakes-beyond-unit
  + 100% test-type coverage mandate (source-of-truth for the column set
  in §2).
- Constitution submodule §11.4.25 (CONST-048) — full-automation
  coverage mandate (defines the six release-gate invariants every PASS
  row must satisfy).
- Constitution submodule §11.4.32 (CONST-055) — post-constitution-pull
  validation sweep (the engine that regenerates this document).
- Constitution submodule §11.4.34 (CONST-058) — Reopened-source
  attribution (the discipline for demoting rows).
- Article XI §11.9 — anti-bluff forensic anchor (the bar every PASS
  cell in §2 must clear).

---

*Last regenerated: 2026-05-19 (round 215 close-out). Next regeneration:
next CONST-055 sweep after constitution-submodule pointer bump.*
