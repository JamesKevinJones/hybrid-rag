# Architecture

Token boundary legend: everything inside `LOCAL COMPUTE` costs electricity.
Everything crossing `=== TOKEN BOUNDARY ===>` costs Claude Pro quota.

The hardware this is shaped by: 6 GB VRAM (about 5.2 GB usable after the Windows
desktop compositor), 16 GB single-channel host RAM (~22 GB/s, half a dual-channel
rig). Every design choice below traces back to one of those two numbers.

```mermaid
flowchart TD
    U[User / MCP client] --> GOV

    subgraph L0["LOCAL COMPUTE - zero cloud tokens"]
        direction TB

        GOV["governance.middleware.ingress<br/>PII scrub -> guardrails -> HMAC-sealed envelope<br/>credential = BLOCKED, not masked"]

        GOV --> RAGQ["rag.local_rag<br/>all-minilm:l6-v2 (384d) -> Qdrant -> TinyBERT rerank<br/>21 ms rerank - emits calibrated confidence"]

        RAGQ --> W{"routing_gate - 3-tier waterfall"}

        W -->|"Tier 1 - 0 ms"| T1["explicit @opus / @local tag<br/>then the regex heuristic table"]
        T1 -->|inconclusive| T2["Tier 2 - 0 ms<br/>retrieval confidence < 0.78 ?"]
        T2 -->|inconclusive| T3["Tier 3 - ~300 ms<br/>ONE token from the RESIDENT 3B<br/>no second model, no eviction"]

        T1 -->|LOCAL| LGEN
        T2 -->|LOCAL| LGEN
        T3 -->|LOCAL| LGEN

        LGEN["llama3.2:3b-instruct-q4_K_M<br/>keep_alive: -1 - PINNED IN VRAM<br/>grounded generation"] --> ANS[Response]
    end

    T1 -->|OPUS| V
    T2 -->|OPUS| V
    T3 -->|OPUS| V

    subgraph VETO["Two vetoes. Either overrules the router."]
        V{"1. envelope verifies?<br/>2. budget window has room?"}
    end

    V -->|"denied"| DEG["degraded_local<br/>recorded, never silent"]
    DEG --> LGEN

    V ==>|"cleared + reserved"| EG["governance.middleware.egress"]
    EG ==> BOUNDARY

    subgraph L2["=== TOKEN BOUNDARY ===  Claude Opus 5"]
        BOUNDARY["Level 2 Oracle"]
        BOUNDARY --> O1[multi-hop decomposition]
        BOUNDARY --> O2[synthetic CoT datasets]
        BOUNDARY --> O3[semantic adjudication]
    end

    O1 --> ANS
    O3 --> ANS
    ANS --> LEDGER
    V -.->|"commit actual spend"| LEDGER

    subgraph MCPL["Module 3 - FastMCP tool server"]
        M1["local_rag_search<br/>every passage screened as CONTEXT"]
        M2[check_budget_status]
        M3[verify_audit_ledger]
    end

    M1 -.-> RAGQ
    M2 -.-> LEDGER
    M3 -.-> LEDGER

    LEDGER[("hash-chained JSONL ledger<br/>latency, tokens, attribution, prompt_hash,<br/>pii_status, guardrail_verdict, fingerprint, seed")]

    subgraph DIST["Modules 4-5 - distillation"]
        COLLECT["distill.dataset_formatter  LOCAL<br/>curate + EGRESS-screen ShareGPT JSONL<br/>credentials dropped, not redacted"]
        COLLECT --> COLAB["scripts/train_colab.py  COLAB T4 15GB<br/>Unsloth 4-bit QLoRA<br/>local VRAM is 1+ GB short"]
        COLAB --> GGUF[("GGUF q4_k_m<br/>ollama create")]
        GGUF -.->|"promote after eval"| LGEN
    end

    O2 --> COLLECT
    LEDGER --> COLLECT
```

## The waterfall, and why the tiers are in that order

| Tier | Cost | Does what | Escalates when |
|---|---|---|---|
| 1a | 0 ms | `@opus` / `@local` tag | the operator said so; outranks everything below |
| 1b | 0 ms | regex heuristic table | score >= 0.75 |
| 2 | 0 ms | calibrated retrieval confidence | confidence < 0.78 |
| 3 | ~300 ms | one token from the resident 3B | it answers `O` |

Tier 3 was a separate 1.5B model until measurement killed it. Two models do not
co-reside in 5.2 GB, so Ollama evicted one per query - a 2-4 second PCIe reload
that cost more than every other stage combined. See decision 14.

## Tier contract

| Tier | Runs on | Owns | Escalates when |
|---|---|---|---|
| Level 0 | Ollama + Qdrant, local | parsing, chunking, embedding, retrieval, rerank, routing, grounded generation | retrieval confidence below threshold |
| Level 1 | local process | tool orchestration, MCP server, budget accounting | a sub-task is flagged high-reasoning |
| Level 2 | Claude Opus 5, cloud | decomposition, synthesis, adjudication | never; it is the terminal authority |

## Governance invariants

1. No payload crosses the token boundary without a verified `GovernanceEnvelope`.
   Enforced in code by `RoutingGate`, not merely documented.
2. Every routing decision is written to the ledger, including the ones that
   stayed local - absence of a cloud call is itself auditable evidence.
3. Every cloud call carries a determinism seed and a model fingerprint.
4. The budget ledger is authoritative and can veto an escalation the router
   asked for. A veto is logged as `degraded_local`, never silently swallowed.
5. Retrieved corpus text is data, never instructions. It is screened as
   `Direction.CONTEXT` and returned quarantined rather than dropped, so a
   poisoned corpus surfaces instead of hiding.
