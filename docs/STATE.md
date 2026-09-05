# State

_Updated 2026-09-03._

## Where we stopped

**Level 2 is no longer stubbed.** Modules 1 through 5 are built, typed, linted
and running. 236 tests pass; ruff and mypy strict are clean.

Shipped this session:

- `llm/opus_client.py` - live Anthropic client behind a `FrontierClient`
  Protocol, plus `UnavailableFrontierClient` as an explicit null object so the
  degradation path is the same code in development as in production.
- `agents/orchestrator.py` - Module 5a. Ingress governance, retrieval on
  redacted text, per-chunk context screening, the routing gate, both execution
  tiers, egress screening of the *response*, distillation harvest, and one
  audit record on every branch including the ones that spent nothing.
- `eval/eval_harness.py` - Module 5b. Groundedness via local embeddings plus
  lexical overlap, instruction precision including negative constraints and
  format tags, and cost per *passed* case. No LLM-as-a-judge anywhere.
- `eval/promotion_gate.py` - Module 5c. GGUF intake, three criteria, and an
  alias update that requires an explicit `apply=True`.
- `scripts/install-wslconfig.ps1` - one-line installer, still not run.
- CLI: `ask`, `evaluate`, `promote`.
- Tests: orchestrator (16), eval harness (30), promotion gate (17), PII edge
  cases (29), audit chain corruption (19), MCP serialization (8 more).

## Verified live

```
hybrid-rag ask "How does the budget veto work?"
  -> local, confidence 0.987, tier1_heuristic, cloud 0+0 tokens

hybrid-rag ask "Compare every distillation curriculum against full fine-tuning"
  -> local_degraded, confidence 0.001, tier2_retrieval_confidence,
     "Level 2 unavailable (no credentials)", cloud 0+0 tokens
```

Cold first call was 51.9 s - that is the one-time model load into VRAM. Warm
calls land around 4.5 s for ~1100 tokens on the 3B, which is `keep_alive: -1`
doing its job.

## Bugs found by writing the tests

1. **Placeholder collision in the PII scrubber.** Scrubbing text that already
   contained `<EMAIL_ADDRESS_1>` minted a second `<EMAIL_ADDRESS_1>`, and
   `restore()` then injected an address where the original had none. This
   system hands its own scrubbed output back to users and MCP callers, so its
   output is a realistic input. Counters now seed above any existing index.
2. **`verify_chain` raised on a corrupt ledger.** A half-written row threw
   `orjson.JSONDecodeError` out of the MCP tool whose entire job is answering
   "is my audit trail sound?". A break is now reported as `(False, line)`.

## Exact next step

The system is complete end to end but has **never called Claude**. To close that:

1. `setx ANTHROPIC_API_KEY ...` in a terminal, or set
   `HYBRID_RAG_ANTHROPIC__API_KEY`. Never paste it into a chat session.
2. `uv run hybrid-rag ask "<a genuinely hard question>"` and confirm the ledger
   shows `tier: opus` with real token counts.
3. Repeat until `data/datasets/distillation_train.jsonl` has a few hundred
   curated samples, then upload it plus `scripts/train_colab.py` to Colab.
4. Build `data/datasets/eval_cases.jsonl` and `opus_baseline.jsonl` from the
   harvested traces so `evaluate` and `promote` have something to score.

## Known gaps

- **Zero cloud tokens have ever been spent.** Every Level 2 test uses a stub.
  The live client is written and typed but unexercised against the real API,
  and `ANTHROPIC_API_KEY` is not set in the environment this runs in.
- Budget limits in `BudgetSettings` are still placeholders (45 msgs / 900k
  tokens / 5h). They must be set from the real account before the first live
  escalation, or the veto guards a number nobody chose.
- No eval cases and no Opus baseline, so `evaluate` and `promote` are proven by
  unit tests and have never scored a real model. Both are downstream of the
  first live call.
- `train_colab.py` has never been run.
- Routing thresholds are calibrated against one 214-chunk corpus of this
  project's own docs - enough to show the signal exists, not enough to claim
  any routing-percentage target.
- The audit chain is unkeyed SHA-256. It detects accidents and casual edits,
  not an attacker who recomputes it. Anchoring it off-machine is unbuilt.
- `scripts/install-wslconfig.ps1` is not run. Machine-wide, and its 6 GB
  ceiling is shared with Docker-in-WSL on this host.
- Presidio is verified and stays OPT-IN: real PERSON/LOCATION recall, but it
  flags "Q3" as an entity, which is destructive on a technical corpus. See
  decision 25.
- CI security review only fires on pull_request. Commits pushed straight to
  main get a `note-unreviewed` warning annotation and nothing else. See
  decision 26 and AGENTS rule 8.
