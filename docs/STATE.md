# State

_Updated 2026-09-05._

## Where we stopped

Modules 1 through 5 are built, calibrated against external data, and running.
282 tests, ruff clean, mypy strict clean. Six commits on `main`.

The system is **air-gapped by default and complete offline**. Level 2 is
optional wiring, not a dependency.

## Shipped this session

- **Budget caps set to deliberate under-estimates**: 25 messages / 400k tokens
  per 5h window, 250 weekly. Not the published Pro limits and not trying to be.
  Guessing low costs a few answers; guessing high costs a lockout mid-session.
- **`.env` key resolution**, which had a real gap: `env_file` maps only
  PREFIXED names onto settings fields and never touches `os.environ`, so an
  ordinary `ANTHROPIC_API_KEY=` line was invisible to the one code path reading
  it. `get_settings()` now loads `.env` first, `override=False`.
- **Air-gap guarantee enforced.** Absent key, malformed key, dead DNS, refused
  handshake, 429, timeout - all arrive as `OpusUnavailable`, all degrade to a
  local answer with zero tokens and the reservation released. New
  `frontier_unavailable` field and compliance flag, because "unreachable",
  "not needed" and "refused" are three different facts.
- **Generator profiles**: `general` (llama3.2:3b) and `code`
  (qwen2.5-coder:3b), plus an explicit override. Unknown profile raises.
- **Threshold recalibrated 0.78 -> 0.35** against an external PEP corpus.
- **Llama Guard chained in** as a content-safety layer, orthogonal to the regex
  injection layer, 38 ms p50, INGRESS/EGRESS only.
- **Presidio verified** for the first time and kept opt-in on measured grounds.
- **AST validation** in the eval harness (`python_syntax` tag) and the
  promotion latency gate moved to **p50 <= 1200 ms** per spec, with p95 still
  reported.
- CI backstop: reverted a fix of mine that produced a green check reviewing
  nothing. See decision 26.

## Measured facts

```
VRAM, all 100% GPU:
  generator llama3.2:3b   2.6 GB     code profile qwen2.5-coder:3b  2.2 GB
  llama-guard3:1b         1.5 GB     llama-guard3:1b                1.5 GB
  all-minilm:l6-v2        0.03 GB    all-minilm:l6-v2               0.03 GB
                          4.1 GB                                    3.7 GB
                                     inside ~5.2 GB usable

Routing separation, external PEP corpus, 665 chunks, 18 labelled queries:
  answerable    n=10  min 0.4984  median 0.9413
  unanswerable  n= 8  max 0.0028  median 0.0000

Guardrail scope:
  regex        catches injection      llama-guard returns `safe` for injection
  llama-guard  catches S1-S13 content regex does not model content harm
```

## Exact next step

**The system has still never called Claude.** Everything downstream of that is
blocked by it, and it is now a two-minute task:

1. Create `.env` in the repo root (gitignored) with `ANTHROPIC_API_KEY=sk-ant-...`
2. `uv run hybrid-rag ask "<a genuinely hard question>"` and confirm the ledger
   shows `tier: opus` with real token counts.
3. Repeat until `data/datasets/distillation_train.jsonl` holds a few hundred
   curated samples, then upload it plus `scripts/train_colab.py` to Colab.
4. Build `data/datasets/eval_cases.jsonl` and `opus_baseline.jsonl` from the
   harvested traces so `evaluate` and `promote` have something real to score.

## Known gaps

- **Zero cloud tokens have ever been spent.** Every Level 2 test uses a stub.
  The live client is written, typed and hardened but unexercised against the
  real API.
- `train_colab.py` has never been run.
- `evaluate` and `promote` are proven by unit tests and have never scored a
  real model - both are downstream of the first live call.
- The calibration set is 18 queries over 8 PEPs. Enough to move a threshold off
  a guessed value with confidence; not enough to claim a routing-percentage
  target. Widen it before quoting one.
- The audit chain is unkeyed SHA-256: it detects accidents and casual edits,
  not an attacker who recomputes it. Anchoring off-machine is unbuilt.
- `scripts/install-wslconfig.ps1` is written and NOT run. Machine-wide, and its
  6 GB ceiling is shared with Docker-in-WSL on this host.
- CI security review only fires on `pull_request`. Direct pushes to main get a
  `note-unreviewed` warning annotation and nothing else.
