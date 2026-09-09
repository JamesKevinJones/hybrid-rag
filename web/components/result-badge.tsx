import type { ExecutionResult, Outcome } from "@/lib/types";

import { Tip } from "./tip";

/**
 * The one-line receipt under an answer.
 *
 * Every field comes off the `ExecutionResult` and nothing is derived here
 * beyond arithmetic on values the pipeline produced. Rule 14: if a display
 * needs a number the pipeline does not produce, the fix is upstream.
 *
 * `cloud 0+0` is deliberately the most prominent cell. It is the project's
 * whole claim, and it is the one number a reader should be able to check at a
 * glance.
 *
 * Every cell explains itself on hover or focus. None of these terms are
 * self-evident, and a receipt nobody can read is decoration.
 */

const OUTCOME_LABEL: Record<Outcome, string> = {
  local: "LOCAL",
  local_degraded: "LOCAL / DEGRADED",
  opus: "OPUS 5",
  blocked: "BLOCKED",
  failed: "FAILED",
};

// Only volt carries white text (4.93:1). Coral looks like it should and does
// not - white on coral is 3.05:1 and fails AA, ink on coral is 6.46:1.
const OUTCOME_STYLE: Record<Outcome, string> = {
  local: "bg-mint text-ink",
  local_degraded: "bg-acid text-ink",
  opus: "bg-volt text-white",
  blocked: "bg-coral text-ink",
  failed: "bg-coral text-ink",
};

const OUTCOME_TIP: Record<Outcome, React.ReactNode> = {
  local: (
    <>
      <strong>Answered on your GPU.</strong> Retrieval confidence cleared the
      threshold, so the routing gate never considered escalating. No cloud tokens
      were spent and none were reserved.
    </>
  ),
  local_degraded: (
    <>
      <strong>It wanted to escalate and could not.</strong> Confidence was below
      the threshold, so the gate chose Level 2 &mdash; then either credentials
      were missing, the budget vetoed it, or the call failed. It fell back to the
      local model rather than erroring.
      <br />
      <br />
      The distinction matters in an audit: &ldquo;Level 2 was unreachable&rdquo;
      and &ldquo;Level 2 was not needed&rdquo; are different facts.
    </>
  ),
  opus: (
    <>
      <strong>This one crossed the token boundary.</strong> Claude Opus 5
      answered, under a budget reservation that was settled afterwards. It
      required a sealed governance envelope, and the response was screened again
      on the way back.
    </>
  ),
  blocked: (
    <>
      <strong>Refused before any model saw it.</strong> Either a guardrail
      matched, or live credential material was detected &mdash; credentials are
      blocked outright rather than masked, because masking a leaked key and
      carrying on normalises the incident.
    </>
  ),
  failed: (
    <>
      <strong>Something broke.</strong> The pipeline never raises; a failure
      becomes a result carrying its reason, so the ledger still records what
      happened.
    </>
  ),
};

function Cell({ children, tip }: { children: React.ReactNode; tip: React.ReactNode }) {
  return (
    <Tip content={tip} className="label whitespace-nowrap">
      {children}
    </Tip>
  );
}

export function ResultBadge({ result }: { result: ExecutionResult }) {
  const tokensOut = result.cloud_tokens_out || result.local_tokens_out;
  const throughput =
    tokensOut > 0 && result.generation_ms > 0
      ? Math.round(tokensOut / (result.generation_ms / 1000))
      : null;

  const pii =
    result.pii_status === "scrubbed"
      ? `PII ${result.pii_entities} REDACTED`
      : `PII ${result.pii_status.replace("_", " ").toUpperCase()}`;

  return (
    <div className="brut-sm mt-4 flex flex-wrap items-center gap-x-5 gap-y-2.5 px-4 py-3">
      <Tip content={OUTCOME_TIP[result.outcome]}>
        <span className={`label px-2.5 py-1 ${OUTCOME_STYLE[result.outcome]}`}>
          {OUTCOME_LABEL[result.outcome]}
        </span>
      </Tip>

      <Cell
        tip={
          <>
            <strong>Cloud tokens in + out.</strong> The project&rsquo;s whole
            claim is that this reads <code>0+0</code> for most traffic. It is
            highlighted because it is the one number a reader should be able to
            check at a glance.
          </>
        }
      >
        <span
          className={
            result.spent_cloud_tokens ? "bg-volt px-1.5 py-0.5 text-white" : "mark"
          }
        >
          cloud {result.cloud_tokens_in}+{result.cloud_tokens_out}
        </span>
      </Cell>

      <Cell
        tip={
          <>
            <strong>Calibrated retrieval confidence.</strong> Below the
            threshold, the corpus genuinely does not contain the answer and the
            gate escalates.
            <br />
            <br />
            Calibrated against an external corpus rather than this
            project&rsquo;s own docs, which would be circular: answerable queries
            scored at least 0.4984, unanswerable ones at most 0.0028.
          </>
        }
      >
        conf {result.retrieval_confidence.toFixed(4)}
      </Cell>

      {/* No denominator, no number. Absent, not zero, not a dash. */}
      {throughput !== null && (
        <Cell
          tip={
            <>
              <strong>Generation throughput.</strong> Output tokens divided by
              model time only &mdash; not wall clock, which also contains
              governance, retrieval and reranking.
            </>
          }
        >
          {throughput} tok/s
        </Cell>
      )}

      <Cell
        tip={
          <>
            <strong>What the scrubber found in your prompt.</strong>{" "}
            <code>clean</code> is nothing detected, <code>scrubbed</code> means
            entities were replaced with placeholders before retrieval or any
            model saw the text, and <code>blocked</code> means live credential
            material, which is refused rather than masked.
            <br />
            <br />
            Redacted values are restored into the answer for you, and never into
            the ledger or the training set.
          </>
        }
      >
        {pii}
      </Cell>

      <Cell
        tip={
          <>
            <strong>End-to-end wall clock</strong> for the whole request:
            governance, retrieval, reranking and generation.
          </>
        }
      >
        {Math.round(result.latency_ms)} ms
      </Cell>

      {result.generation_ms > 0 && (
        <Cell
          tip={
            <>
              <strong>Model time alone.</strong> The gap between this and total
              latency is what the pipeline around the model costs.
            </>
          }
        >
          gen {Math.round(result.generation_ms)} ms
        </Cell>
      )}

      {result.quarantined_chunks > 0 && (
        <Cell
          tip={
            <>
              <strong>A retrieved passage tried to issue instructions.</strong> It
              was dropped from the prompt but reported rather than silently
              discarded &mdash; you need to know the corpus is compromised.
              Retrieved text is data, never instructions.
            </>
          }
        >
          <span className="bg-coral px-1.5 py-0.5 text-ink">
            {result.quarantined_chunks} QUARANTINED
          </span>
        </Cell>
      )}

      {result.audit_hash && (
        <Cell
          tip={
            <>
              <strong>This turn&rsquo;s row in the audit ledger.</strong> Every
              record chains to the one before it, so an edited row and a deleted
              row are both detectable.
              <br />
              <br />
              Open the ledger drawer below, or run{" "}
              <code>hybrid-rag verify-ledger</code> to re-walk the whole chain.
            </>
          }
        >
          <span className="opacity-75">
            ledger {result.audit_hash.slice(0, 8)}&hellip;
          </span>
        </Cell>
      )}
    </div>
  );
}
