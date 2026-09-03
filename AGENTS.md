# hybrid-rag

Canonical context. Claude Code, Antigravity and Codex all read this file.
Session status lives in `docs/STATE.md`, rationale in `docs/DECISIONS.md`.

## What this is

A tiered, local-first multi-agent RAG framework. Level 0 (Ollama + a local
vector index) answers everything it credibly can; Level 2 (Claude Opus 5) is
reserved for query decomposition, synthetic dataset generation and semantic
adjudication. A single `RoutingGate` decides which tier answers, a rolling
window budget can veto that decision, and every decision is appended to a
hash-chained governance ledger. Downstream, harvested Opus traces feed a QLoRA
distillation loop that trains a local model to take over more of the work.

The point is not "use a local model." It is: spend cloud quota only on work a
local model provably cannot do, and be able to prove afterwards which was which.

## Stack

- Python 3.12 (pinned `<3.13`; torch/bitsandbytes/unsloth wheels lag). System
  interpreter is 3.14, so always go through `uv run`, never bare `python`.
- uv for env and lockfile. `uv.lock` is the strict lock; `pyproject.toml`
  carries bounded ranges.
- pydantic v2 + pydantic-settings for typed config and DI.
- httpx + tenacity for the Ollama transport. Async throughout.
- qdrant-client in embedded (`path=`) mode by default; chromadb behind the same
  `VectorStore` protocol via the `chroma` extra.
- `fastmcp` 2.x for the MCP tool server.
- FlashRank (`ms-marco-TinyBERT-L-2-v2`) in CORE deps, not an extra. Without
  a cross-encoder the routing confidence signal is noise.
- No local training. QLoRA runs on a Colab T4; see `scripts/train_colab.py`.

## Layout

```
src/hybrid_rag/
  config.py          # every tunable; nothing else reads os.environ
  container.py       # composition root - the ONLY place backends are constructed
  cli.py             # thin operator commands over the container
  routing/
    routing_gate.py  # Module 1a - the token boundary
    budget.py        # rolling-window ledger, holds a hard veto over the gate
  rag/
    local_rag.py     # Module 1b - chunking, vector stores, rerank, confidence
  governance/
    audit.py         # AuditRecord schema + hash-chained JSONL sink
    pii.py           # Module 2a - detectors, redaction, credential blocking
    guardrails.py    # Module 2b - regex + Llama-Guard, ingress/context/egress
    middleware.py    # Module 2c - GovernanceEnvelope, enforces invariant 1
  llm/
    ollama_client.py # Level 0 transport
  mcp_servers/
    server.py        # Module 3 - FastMCP: search, budget, ledger tools
  distill/
    dataset_formatter.py  # Module 4a - curation. Training is remote, on purpose.
  agents/            # not built yet
scripts/
  train_colab.py     # Module 4b - runs on Colab, imports nothing from this package
  wslconfig.template # WSL2 memory caps; NOT installed automatically
```

## Rules

1. Constructor injection only. If a module imports a concrete backend, that
   import belongs in `container.py` instead.
2. The fail-safe direction is always LOCAL. A dead Ollama, an unparseable SLM
   vote, a corrupt ledger - none of them may cost cloud tokens.
3. No decision goes unlogged, including the ones that stayed local. Absence of
   a cloud call is auditable evidence.
4. Never write raw prompt text to the ledger - `hash_prompt()` only. That
   includes guardrail finding evidence, which is scrubbed before storage.
5. Nothing crosses the token boundary without a verified `GovernanceEnvelope`.
   `require_envelope=False` is an auditable downgrade, not a performance knob.
6. One model resident at a time. Passing an explicit `model=` to the Ollama
   client is what evicts it - do it deliberately or not at all.
7. `relevance_floor` is a property of the cross-encoder. Re-measure it whenever
   `rerank_model` changes; a stale floor silently breaks routing.
8. Run everything in `docs/VERIFY.md` before reporting work done.

## Working with the token budget

This project is built under Claude Pro message caps, so the interaction style
is deliberate: one megaprompt per module rather than a conversational drip,
a fresh session per module, and error reports trimmed to the failing function
plus the specific traceback line rather than a pasted 200-line stack.

## Don't touch

- `data/` - generated indexes, ledgers and datasets. Gitignored.
- `uv.lock` - regenerate with `uv lock`, do not hand-edit.
