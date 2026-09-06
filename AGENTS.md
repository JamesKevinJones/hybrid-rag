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
- Measured VRAM envelope, 100% GPU: generator 2.6 + llama-guard3:1b 1.5 +
  all-minilm 0.03 = **4.1 GB** inside ~5.2 GB usable. The `code` profile
  (qwen2.5-coder:3b) totals 3.7 GB. Every generator profile must be ~3B Q4;
  a 7B breaks the envelope silently by spilling to shared system memory.
- Level 2 is optional. `ANTHROPIC_API_KEY` in a gitignored `.env` enables it;
  absent, the system is fully air-gapped and every escalation degrades.

## Layout

```
src/hybrid_rag/
  config.py          # every tunable; nothing else reads os.environ
  container.py       # composition root - the ONLY place backends are constructed
  cli.py             # thin operator commands; bare invocation opens the TUI
  ui/
    tui.py           # Module 6 - interactive terminal UI. Presentation only.
  routing/
    routing_gate.py  # Module 1a - the token boundary
    budget.py        # rolling-window ledger, holds a hard veto over the gate
  rag/
    local_rag.py     # Module 1b - chunking, vector stores, rerank, confidence
  governance/
    audit.py         # AuditRecord schema + hash-chained JSONL sink
    pii.py           # Module 2a - detectors, redaction, credential blocking
    guardrails.py    # Module 2b - regex (injection) + Llama-Guard (content safety)
    middleware.py    # Module 2c - GovernanceEnvelope, enforces invariant 1
  llm/
    ollama_client.py # Level 0 transport
    opus_client.py   # Level 2 transport - the only file that can spend money
  mcp_servers/
    server.py        # Module 3 - FastMCP: search, budget, ledger tools
  distill/
    dataset_formatter.py  # Module 4a - curation. Training is remote, on purpose.
  agents/
    orchestrator.py  # Module 5a - the live execution path, both tiers
  eval/
    eval_harness.py  # Module 5b - local metrics, replayed baseline, no judge model
    promotion_gate.py # Module 5c - GGUF intake, 3 criteria, opt-in alias update
scripts/
  train_colab.py     # Module 4b - runs on Colab, imports nothing from this package
  verify_mcp_clients.py # real stdio JSON-RPC handshake against the MCP server
  wslconfig.template # WSL2 memory caps; NOT installed automatically
  install-wslconfig.ps1 # one-liner; machine-wide, shared with Docker-in-WSL
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
6. Passing an explicit `model=` to the Ollama client loads a second model. Do
   it deliberately. It does NOT necessarily evict the generator - three models
   co-reside here at 100% GPU - but each one costs VRAM you have not budgeted.
7. `relevance_floor` is a property of the cross-encoder. Re-measure it whenever
   `rerank_model` changes; a stale floor silently breaks routing.
8. CI security review only runs on pull_request - the action skips every other
   event. Work pushed straight to main is covered by the local
   /security-review command and nothing else. Open a PR to get CI coverage.
9. A budget or governance veto makes ZERO cloud calls. Not cancelled, not
   discarded - the frontier client is never touched. There is a test for it.
10. Restored PII goes to the user only. Never into the ledger, never into the
   training set. `ScrubResult.reverse_map` is excluded from serialization.
11. Two guardrail layers, ORTHOGONAL not stacked. Regex = prompt injection,
    exfiltration, residual secrets. Llama Guard = harmful content only; it
    returns `safe` for a textbook injection. Neither substitutes for the other.
12. Thresholds are calibrated against `data/calibration` (external PEPs), never
    against this repo's own docs - that is circular. Re-run
    `scripts/calibrate_thresholds.py` after any reranker or embedder change.
13. Run everything in `docs/VERIFY.md` before reporting work done.
14. Nothing in `ui/` decides, spends or logs. It renders what the pipeline
    produced. If a display needs a number the pipeline does not produce, add it
    to `ExecutionResult` - do not compute a plausible one in the renderer.
15. Every value the UI shows is live, cited, or absent. A recorded benchmark is
    labelled with the decision that recorded it. "Not measured" is a valid
    status; a placeholder that looks like a reading is not. See decision 29.

## IDE integration

- `.vscode/` - tasks, pinned interpreter, `mcp.json`. VS Code is installed here.
  The server launches via `${workspaceFolder}/.venv/Scripts/python.exe -m
  hybrid_rag.mcp_servers.server`. NOT `src.hybrid_rag...` - that form resolves
  only from the repo root and imports the package twice when it does. Decision 30.
- `.antigravity/instructions.md` - the load-bearing Antigravity file; plain
  markdown any agent can read. `antigravity.json` beside it has an UNVERIFIED
  schema (agy 1.1.12 exposes no `mcp` subcommand) and says so in the file.
- No Cursor config: Cursor is not installed on this machine.

The MCP server registers as **`hybrid-rag`** with tools `local_rag_search`,
`check_budget_status`, `verify_audit_ledger`. Earlier drafts said
`LocalGovEngine` and `search_local_knowledge`; those names do not exist and
calling them fails. They are frozen by decision 28 - a mismatch is a stale
client config, not a rename to apply.

Check the whole chain with `uv run python scripts/verify_mcp_clients.py`: it
runs a real stdio JSON-RPC session against the server from a foreign working
directory and validates both editor configs on the way.

## Working with the token budget

This project is built under Claude Pro message caps, so the interaction style
is deliberate: one megaprompt per module rather than a conversational drip,
a fresh session per module, and error reports trimmed to the failing function
plus the specific traceback line rather than a pasted 200-line stack.

## Don't touch

- `data/` - generated indexes, ledgers and datasets. Gitignored.
- `uv.lock` - regenerate with `uv lock`, do not hand-edit.
