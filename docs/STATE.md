# State

_Updated 2026-09-03._

## Where we stopped

Modules 1 through 4a are built and running against live local models. Module 3
(MCP) and Module 4b (Colab training) are new this session. 116 tests pass;
ruff and mypy strict are clean.

## What changed this session: the hardware refactor

Everything below traces to two numbers - ~5.2 GB usable VRAM and ~22 GB/s of
single-channel memory bandwidth.

- **Deleted the 1.5B router.** Two models do not co-reside in 5.2 GB, so Ollama
  evicted one per query: a 2-4 second PCIe reload that cost more than every
  other pipeline stage combined. Tier 3 now asks the resident 3B for a single
  token. `keep_alive: -1` on every request pins it, asserted at the wire level.
- **Three-tier waterfall** with explicit `@opus` / `@local` tags at tier 1a.
- **Embeddings moved to `all-minilm:l6-v2`** - 384 dimensions instead of nomic's
  768, halving the resident index. The collection name encodes the dimension so
  a model change fails loudly instead of returning confident nonsense.
- **Reranker moved to `ms-marco-TinyBERT-L-2-v2`** - 21 ms p50 versus 552 ms for
  MiniLM-L-12. 26x, and the reason to switch turned out to be latency rather
  than the RAM it was requested for.
- **`relevance_floor` is now config**, set to 0.05 from measured TinyBERT score
  distributions. This is the trap: the floor is a property of the cross-encoder,
  and the old 0.5 made real questions score 0.000 after the swap.
- **`max_workers: 2`** as semaphores in both the Ollama client and the RAG
  engine, with a leak test.
- **Module 3**: `mcp_servers/server.py` on FastMCP 2.x. Three tools, none able
  to spend cloud quota. Retrieved passages are screened as `Direction.CONTEXT`
  and returned `quarantined` rather than dropped.
- **Module 4a**: `distill/dataset_formatter.py`. Local curation with egress
  screening; credentials drop the sample rather than being redacted.
- **Module 4b**: `scripts/train_colab.py`. Standalone, T4, Unsloth QLoRA, GGUF
  export straight back into `ollama create`.
- **`scripts/wslconfig.template`** - written, deliberately NOT installed.

## Bugs found by running it

1. `top=0.002, margin=0.871` - a margin between two near-zero scores is
   numerically huge and semantically empty, and a junk query rode it to 0.35
   confidence. Margin now only counts above the relevance floor.
2. A missing Qdrant collection raised a raw `ValueError` out of an MCP tool
   call. An index that was never built is an empty index, not an error.

## Exact next step

Module 5 - the evaluation harness (`src/hybrid_rag/distill/eval_harness.py`):

1. Ragas or G-Eval scoring of the local pipeline against recorded Opus
   completions from the ledger.
2. A promotion gate: the distilled model replaces the base only if the score
   gap is under a configured delta. Without this, `train_colab.py` produces an
   adapter nobody can justify deploying.
3. A labelled query set. The routing thresholds are calibrated against one
   214-chunk corpus of this project's own docs - enough to prove the signal
   exists, not enough to claim any routing-percentage target.

Then Module 4c: the Opus client and cloud-agent bridge (`llm/opus_client.py`,
`agents/cloud_bridge.py`), which is the only remaining piece that actually
spends tokens.

## Known gaps

- **No component has ever called Claude.** The Level 2 path is fully specified,
  budgeted, governed and tested at its boundary - and stubbed. Nothing in this
  repo has spent a single cloud token.
- `train_colab.py` has never been run. It is syntax-checked, not executed.
- `llama-guard3:1b` is pulled but not chained in by default: loading it evicts
  the generator, which is exactly what decision 14 was about.
- The Presidio detector is written but untested against a live spaCy model.
- `scripts/wslconfig.template` is not installed. It caps WSL2 at 6 GB, which is
  shared with Docker-in-WSL on this host - applying it without checking
  container memory first will OOM-kill them.
- Budget limits in `BudgetSettings` are still placeholders.
