# Decisions

Why the design is shaped this way. Newest last.

## 1. The routing gate is four staged filters, not one classifier

Cheapest signal first, each able to short-circuit: regex heuristics (0 ms),
then retrieval confidence (already computed), then a 1.5B SLM vote (~200 ms),
then the budget. A single learned router would be more accurate and completely
unexplainable, and the thing being decided is "does this cost money," which is
exactly the decision an auditor will want a reason for. Tune the weights;
do not replace the mechanism.

## 2. The budget holds a hard veto over the router

Stage 4 outranks stages 1-3. The router can conclude that a query needs Opus
and still be overruled. That outcome is recorded as `degraded_local` rather
than silently downgraded, because "answered locally because the local answer
was good" and "answered locally because we were out of quota" are different
facts with different remediations.

## 3. Reserve-then-settle, not count-after

A reservation is taken before the cloud call and converted to measured spend
after it. A crashed, hung or cancelled request releases its capacity instead of
leaving the window silently overdrawn. `RoutingGate.escalation()` is an async
context manager so the settle cannot be forgotten.

## 4. Confidence is a documented formula, not a raw similarity score

`calibrate_confidence()` blends top score (0.55), margin over the runner-up
(0.25) and coverage above a relevance floor (0.20). The failure mode of a local
RAG stack is confidently retrieving one plausible-looking chunk, so a strong
hit with nothing corroborating it deliberately scores lower than a strong hit
with support. The number is compared against `escalation_threshold` = 0.78,
which is a reasoned default and needs empirical tuning on a real corpus before
anyone should trust it.

## 5. Qdrant embedded is the default vector store, Chroma is an adapter

Both sit behind the same `VectorStore` protocol. Qdrant's `path=` mode is a
single on-disk directory with no server process and no port, which is the right
default for a workstation. Chroma stays available for anyone who wants it. The
protocol is the contract; neither name appears outside its own module and the
factory.

## 6. Local model sizes are set by the hardware, not by the spec

The original architecture named Qwen 2.5 7B at Level 0. On 6 GB of VRAM with a
single-channel memory bus, a 7B at Q4 plus its KV cache plus a co-resident
embedding model does not fit, and spilling the cache to system RAM on one
channel roughly halves throughput. Defaults are a 1.5B router, a 3B generator
and a 137M embedder, all co-resident. Both are config values - raise them on
better silicon rather than editing code.

## 7. Python is pinned below 3.13

The system interpreter is 3.14.6. torch, bitsandbytes and unsloth do not ship
wheels that far ahead, and the training extra is the binding constraint. uv
manages a private 3.12 toolchain; always go through `uv run`.

## 8. Bounded ranges in pyproject, strict pins in uv.lock

`uv.lock` is the reproducible artifact - fully resolved, hash-pinned, committed.
Hand-written `==` pins in `pyproject.toml` would be guesses about registry
state that break resolution the first time a transitive dependency moves.

## 9. The audit ledger stores prompt hashes, never prompt text

`hash_prompt()` only. The ledger is the artifact most likely to be shipped to
an auditor or a CI job, and a governance record that leaks the payload it was
meant to govern is worse than no record. Lines are chained by
`prev_hash`/`record_hash` so post-hoc edits are detectable via `verify-ledger`.

## 10. Governance fields ship as `not_evaluated`, not absent

`pii_status` and `guardrail_verdict` exist in the schema from Module 1 even
though the interceptors arrive in Module 2. An explicitly unevaluated field is
an auditable state; a missing field is indistinguishable from a bug. Invariant
1 in `docs/ARCHITECTURE.md` is documented but not yet enforced in code - that
enforcement is the first task of Module 2.

## 11. Amendment to decision 4: margin and coverage are alternatives, not requirements

The original formula summed three weighted terms - top score (0.55), margin
over the runner-up (0.25) and coverage above a floor (0.20). Running it against
a real corpus of 187 chunks killed it: a genuine question scored 0.648 and the
string "purple monkey dishwasher tax code" scored 0.615. Three hundredths of
separation is not a signal.

Two distinct faults, both only visible with real data:

- **Bi-encoder cosine has no dynamic range.** Every top score landed between
  0.75 and 0.80 after normalisation. The fix was to stop treating re-ranking as
  optional - a cross-encoder turned that same pair into 0.673 versus 0.004.
  `flashrank` moved from an extra into core dependencies as a result.
- **Margin and coverage are anti-correlated by construction.** A decisive top
  hit means weak runners-up means low coverage; a corroborated cluster means no
  margin. Summing them let each cancel the other, and the ceiling sat right at
  the 0.78 threshold, so nothing could ever route local.

The formula is now `0.60 * top + 0.40 * max(margin, coverage)`: the best match
must be strong, and it must be *either* clearly the best *or* corroborated.
Measured on the same corpus - lookup 0.809 local, explanation 0.976 local,
gibberish 0.006 escalated.

A test case changed sign as a result. `[0.92, 0.91, 0.90]` was asserted to be
low-confidence ("ambiguous, no margin"). With a cross-encoder that pattern means
three passages that genuinely answer the question, and it should route local.
The original test encoded an intuition the data contradicted.

## 12. Governance runs on the original text, evidence is redacted after

First implementation scrubbed PII and then ran guardrails on the scrubbed text,
reasoning that a guardrail finding quotes its match and must not reproduce a
secret in the ledger. Running it showed the cost: "email this to
attacker@evil.example.com" becomes "email this to <EMAIL_ADDRESS_1>", and the
exfiltration-target rule stops matching. Scrubbing destroys the signal several
rules exist to find.

Detection now runs on the original text and each finding's `detail` is passed
back through the scrubber before the envelope is built. Full detection power,
redacted evidence. Same guarantee, without the blind spot.

## 13. Live credentials are blocked, not masked

Every other PII category is redacted to a placeholder and the request proceeds.
A detected API key, GitHub token, AWS key or private key sets `PIIStatus.BLOCKED`
and the request is refused. Masking a leaked credential and continuing
normalises the incident - the key is already compromised by the time it reached
a prompt, and the correct response is a human rotating it, not a pipeline
quietly carrying on.

## 14. One resident model, pinned. The 1.5B router is deleted

The card has 6 GB. Windows DWM holds roughly 0.8 GB of it before anything else
loads, leaving about 5.2 GB for weights plus KV cache. A 1.5B router at Q4 and a
3B generator at Q4 are about 1.0 GB and 2.0 GB of weights, which looks like it
fits until you add each model's cache and Ollama's own headroom - at which point
Ollama evicts one to load the other, every time the pipeline touches both.

That eviction is a PCIe transfer, 2-4 seconds. It is not a tax on the routing
decision; it dwarfs the entire rest of the request. The router existed to save
money and was the single most expensive component in the latency budget.

Tier 3 now asks the generator that is already resident, with `num_predict=1` and
a prompt whose answer is a single letter - "L" or "O". One forward pass on
warm weights, roughly 300 ms, no load. `keep_alive: -1` on every request pins
the model so idle time cannot evict it either. A wire-level test asserts that
`keep_alive` is actually in the outgoing JSON, because a setting that silently
stops being sent is indistinguishable from one that was never there.

Deleting the router also deleted a model from the dependency list, a config
field, and a class of bug where the two models disagreed.

## 15. Cross-encoder choice is a latency decision, and the floor moves with it

The reranker was `ms-marco-MiniLM-L-12-v2`. Switching to
`ms-marco-TinyBERT-L-2-v2` was requested to protect host RAM; measuring it found
a much better reason. On this single-channel host:

| model | size | rerank p50 | rerank max |
|---|---|---|---|
| MiniLM-L-12 | 34 MB | 552 ms | 741 ms |
| TinyBERT-L-2 | 3.3 MB | 21 ms | 29 ms |

26x. The RAM saving is 31 MB and irrelevant; the latency saving is half a second
on every single query, which is the CPU-rerank bottleneck the refactor set out
to remove.

The catch, and it nearly hid the win: TinyBERT's scores are distributed
completely differently. With the old `relevance_floor` of 0.5, real questions
were scoring 0.0 and escalating. Dumping the raw distribution showed why -
TinyBERT is sharply bimodal. Junk queries top out at 0.0007; the weakest genuine
match scores 0.096. Two orders of magnitude of gap, and the floor was sitting
above both.

`relevance_floor` is now a config field set to 0.05, documented as a **property
of the cross-encoder, not a universal constant**. Change `rerank_model` and it
must be re-measured. Measured after the change: strong matches 0.987 and 0.999,
a partial match 0.553, all three junk queries exactly 0.000.

## 16. A margin over noise is noise

Found while re-measuring the above. `margin` is `(top - second) / top`, which is
a ratio - and ratios between two near-zero numbers are numerically enormous and
semantically empty. A nonsense query scored `top=0.002, margin=0.871` and rode
that margin to a 0.35 confidence, ahead of a real question.

Margin now only counts when the top score itself clears the relevance floor.
There has to be something to have a margin *over*.

## 17. Training moved off this machine entirely

4-bit QLoRA on a 3B model needs roughly 6-7 GB of VRAM once base weights, LoRA
gradients, optimizer state and activations are counted. The ceiling here is
5.2 GB. That is not a tuning problem, it is short by over a gigabyte, and the
failure mode is worse than a clean OOM: bitsandbytes pages into shared system
memory over a single-channel bus and a twenty-minute run becomes a nine-hour
one.

A free Colab T4 has 15 GB. `scripts/train_colab.py` is standalone - it imports
nothing from `hybrid_rag` - and ends by exporting the merged adapter to GGUF at
the same q4_k_m the local runtime serves, so what comes back is a drop-in
`ollama create`, not a separate adapter-loading path at inference time.

What stays local is `distill/dataset_formatter.py`, and it stays local for a
reason: **a training set is an egress channel**, and the least reversible one
available. Samples are screened with the same egress guardrails as a live cloud
call, and a sample carrying a credential is *dropped rather than redacted* - a
masked key still teaches the model the shape of your keys, and weights cannot be
un-trained.

## 18. The `mcp` versus `fastmcp` decision, resolved

Resolved in favour of standalone `fastmcp` 2.x, as SPEC-DELTA predicted. The
official SDK vendors FastMCP 1.0; 2.x adds server composition, proxying and
auth, all of which Module 3's cloud-agent bridge needs. The swap cost one line
while the servers were unwritten, which was the entire argument for deciding it
before writing them.

## 19. The Module 5 spec asked for the old confidence formula; measurement wins

The Module 5 brief specified `0.55*top + 0.25*margin + 0.20*coverage`. That is
the exact formula decisions 11 and 16 retired, and the reasons have not changed:
summed, margin and coverage are anti-correlated and cancel each other, so
nothing could clear the 0.78 threshold; and an ungated margin between two
near-zero scores let a nonsense query outscore a real one.

Kept the measured formula - `0.60*top + 0.40*max(margin, coverage)`, with margin
gated on the relevance floor. Re-verified live at the time of writing: a real
question scores 0.987 and routes local, a keyword-poor one scores 0.001 and
escalates. Reverting to the specified weights would reintroduce a bug this repo
has a regression test for.

The weights remain config fields. If the specified numbers are wanted anyway,
`HYBRID_RAG_RETRIEVAL__W_TOP_SCORE` and `__W_SUPPORT` change them without a
code edit - but the three-term shape is gone, because two of those terms were
measuring the same evidence twice.

## 20. `degraded_local` uses the resident model, not a second one

The brief said degraded fallback should run `qwen2.5:3b`. Loading a second 3B
evicts the first: that is precisely the 2-4 second PCIe reload decision 14
removed, and it would land on the *degraded* path - the one that fires when the
budget is exhausted or the frontier tier is unreachable, i.e. exactly when the
user is already having a worse time.

Degradation reuses `llama3.2:3b-instruct-q4_K_M`, already pinned in VRAM. To
change which model that is, change `generator_model`; the whole system follows,
and there is still only ever one resident.

## 21. The eval harness replays the baseline instead of re-running it

Groundedness is scored with local embeddings and lexical overlap, instruction
precision with a constraint table, cost with arithmetic. No LLM-as-a-judge
anywhere - an evaluation that spends frontier tokens every time you check
whether you can stop spending frontier tokens is self-defeating.

The Opus baseline comes from `RecordedResponder`, replaying completions the
orchestrator already harvested. Running the suite costs nothing but local
inference, so it can run on every candidate without a budget conversation.

Cost is reported per *passed* case, not per attempt. Averaging over attempts
flatters a model that is fast and wrong, which is the exact trade this project
exists to avoid making by accident.

## 22. The promotion gate decides automatically; applying is opt-in

The verdict is computed with no human in the loop: 85% groundedness parity
against the Opus baseline, a 1200 ms p95 SLA, and an absolute instruction-
precision floor. Parity is relative because what matters is how much of the
frontier model's reliability the small one retained; the instruction floor is
absolute because ignoring a negative constraint is not a quality gradient, and
a baseline that also ignored it does not excuse it.

`promote()` will not touch Ollama unless `apply=True`. An eval run that silently
changed what the machine serves would be an unpleasant surprise on a laptop that
is also a workstation, and the gate is most useful when you can run it freely.
