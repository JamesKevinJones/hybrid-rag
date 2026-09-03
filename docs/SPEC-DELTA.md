# Delta against the reference design

_Checked 2026-09-03 against the pasted Module 0-3 reference implementation._

The reference sketch and this repo agree on the architecture. They disagree on
several implementation details, and in six places the reference has a defect
this repo does not. Recorded here so the divergence is a decision, not a drift.

## Where this repo is deliberately different

| Topic | Reference | Here | Why |
|---|---|---|---|
| Package shape | flat `src/*.py` | `src/hybrid_rag/` with subpackages | installable, importable, `uv run hybrid-rag` entry point |
| Session directives | in `CLAUDE.md` | in `AGENTS.md` (`CLAUDE.md` is `@AGENTS.md`) | only Claude Code reads `CLAUDE.md`; Codex and Antigravity would not see it |
| `requires-python` | `>=3.11` | `>=3.12,<3.13` | unbounded resolves to 3.14 here and torch has no wheel; the training extra is the constraint |
| Embeddings | `sentence-transformers` in core deps | `nomic-embed-text` over Ollama | sentence-transformers pulls torch into the base install and runs embeddings in-process, competing for RAM with the model server on a single-channel 16 GB box |
| Vector store | ChromaDB, concrete | `VectorStore` Protocol, Qdrant default, Chroma adapter | swappable without touching callers; embedded Qdrant needs no server process |
| Local models | `qwen2.5:3b` router | 1.5B router, 3B generator, 137M embedder | 6 GB VRAM; a 7B-Q4 plus KV cache plus embedder does not co-reside |
| MCP library | `fastmcp>=2.0` | `fastmcp>=2.0` | **resolved in the reference's favour** - see below |
| Ledger contents | stores `sanitized_query` text | stores `prompt_hash` only | defensible either way; scrub recall is not 100 percent, and the ledger is the artifact most likely to leave the machine |

## Defects in the reference this repo does not have

1. **No budget enforcement anywhere.** Constraint 3 of the brief is the Claude
   Pro rolling window; nothing in the reference counts messages or tokens, so
   the router can escalate until the account is exhausted. Here that is
   `routing/budget.py`, and it holds a veto over the router.
2. **No chunking.** `ingest_document` upserts an entire document as a single
   embedding. A 40-page PDF becomes one vector, which destroys retrieval
   precision. Here: `SemanticChunker`, sentence-aligned with overlap.
3. **No re-ranking.** The brief's Level 0 names FlashRank; the reference has
   neither the dependency nor the call. Here: `FlashRankReranker` behind a
   Protocol, degrading to `NullReranker` when the extra is absent.
4. **`similarity_score = 1.0 - dist` can go negative.** Chroma's cosine space
   returns distance in [0, 2], so that expression is cosine similarity in
   [-1, 1], not a [0, 1] score - it cannot be thresholded as one. Here every
   score is normalised to [0, 1] at the adapter boundary, and confidence is a
   documented blend of top score, margin and coverage.
5. **The audit hash does not chain.** `integrity_signature` covers one record,
   so editing a line is detectable but *deleting a whole line is not*. Here
   each record carries `prev_hash`, and `verify-ledger` names the first break.
6. **Module 2 has no guardrails.** The brief requires regex plus a
   Llama-Guard-class filter before any cloud dispatch; the reference ships PII
   scrubbing only. Here: `RegexGuardrail` (always on) plus an optional
   `LlamaGuardGuardrail`, with `Direction.CONTEXT` for indirect injection.

Two smaller ones: `AnalyzerEngine()` is constructed unconditionally, but
presidio needs a spaCy model that is not in the dependency list, so the
middleware fails to instantiate on a fresh install; and `SemanticRouter` uses a
blocking `httpx.Client` inside what the brief describes as an async agent
framework.

## Resolved decision: `mcp` or `fastmcp`

**Resolved 2026-09-03, in the reference's favour.** The official `mcp` SDK
vendors FastMCP 1.0; standalone `fastmcp` 2.x is a superset with server
composition, proxying and auth, all of which Module 3's cloud-agent bridge
needs. Swapped before any server code was written, which cost one line - the
entire argument for deciding it early.

Also worth recording: the reference's `sentence-transformers` embedding choice
was rejected here for pulling torch into the base install, and that reasoning
still holds. But the embedding model itself moved to `all-minilm:l6-v2` - the
same architecture the reference named - served through Ollama rather than
in-process. 384 dimensions instead of nomic's 768 halves the resident index.
