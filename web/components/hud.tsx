import type { Budget, Health } from "@/lib/types";

import { Tip } from "./tip";

/**
 * The status strip, pinned under the header.
 *
 * It lives in the chrome rather than in the empty state for two reasons. It
 * used to be four stacked cards inside the welcome screen, which pushed the
 * heading off the first viewport - you landed on a status panel instead of the
 * thing you came to use. And air-gap status is worth seeing *while* you are
 * asking questions, not only before the first one.
 *
 * Every cell is a tooltip trigger. None of these four terms mean anything to a
 * visitor who has not read the repo, and a status bar full of unexplained
 * jargon is decoration.
 *
 * Two recorded benchmarks appear here (21 ms rerank, 38 ms guard) and both name
 * the decision that recorded them, so a reader can tell a citation from a live
 * reading. Everything else is measured at request time, and "not measured" is a
 * valid state that renders as itself - decision 29.
 */

function Cell({
  label,
  value,
  note,
  tip,
  tone = "plain",
}: {
  label: string;
  value: string;
  note?: string;
  tip: React.ReactNode;
  tone?: "plain" | "good" | "warn" | "bad";
}) {
  const accent =
    tone === "good"
      ? "bg-mint text-ink"
      : tone === "warn"
        ? "bg-acid text-ink"
        : tone === "bad"
          ? "bg-coral text-ink"
          : "";
  return (
    <Tip
      content={tip}
      className="flex min-w-[13rem] shrink-0 flex-col justify-center gap-0.5 border-r-[3px] border-[var(--line)] px-4 py-2.5 last:border-r-0 sm:min-w-0 sm:flex-1"
    >
      <span data-anim="cell" className="flex flex-col gap-0.5">
        <span className="label text-[0.6875rem] opacity-75">{label}</span>
        <span
          className={`truncate font-mono text-[0.875rem] font-medium ${
            accent ? `${accent} w-fit max-w-full px-1.5` : ""
          }`}
        >
          {value}
        </span>
        {note && <span className="truncate text-[0.75rem] opacity-75">{note}</span>}
      </span>
    </Tip>
  );
}

export function Hud({
  health,
  budget,
}: {
  health: Health | null;
  budget: Budget | null;
}) {
  if (!health) {
    return (
      <div className="border-b-[3px] border-[var(--line)] px-4 py-3 sm:px-6">
        <span className="label opacity-75">status</span>{" "}
        <span className="font-mono text-[0.875rem]">reading&hellip;</span>
      </div>
    );
  }

  const budgetValue = budget
    ? `${budget.messages_used}/${budget.messages_limit} msgs`
    : "not read";

  return (
    <div className="border-b-[3px] border-[var(--line)]">
      {/* Horizontal scroll on narrow screens rather than a wrap: a status bar
          that changes height as the window narrows makes the composer jump. */}
      <div className="mx-auto flex w-full max-w-3xl overflow-x-auto">
        <Cell
          label="model"
          value={health.generator}
          note={health.residency}
          tone={health.generator_resident ? "good" : "plain"}
          tip={
            <>
              <strong>The local generator.</strong> A ~3B model quantised to 4
              bits, answering on your own GPU. &ldquo;Resident&rdquo; means the
              weights are already in VRAM, so a question costs no load time;
              otherwise the first one waits about a minute.
              <br />
              <br />
              It is pinned with <code>keep_alive: -1</code> so an idle timeout
              cannot silently unload it. {health.embed_model} handles embeddings
              at {health.embed_dim} dimensions.
            </>
          }
        />
        <Cell
          label="reranker"
          value={health.rerank_enabled ? health.rerank_model : "disabled"}
          note="21 ms p50, decision 15"
          tone={health.rerank_enabled ? "plain" : "warn"}
          tip={
            <>
              <strong>A cross-encoder that re-scores retrieved passages.</strong>{" "}
              Without it the confidence number is noise: bi-encoder similarity
              alone separates a real question from gibberish by about 0.03.
              <br />
              <br />
              TinyBERT was chosen over MiniLM-L-12 on latency, not quality &mdash;
              21 ms against 552 ms on this hardware, a 26&times; difference on
              every single query.
            </>
          }
        />
        <Cell
          label="guardrails"
          value={health.guard_enabled ? `${health.guard_model} + regex` : "regex only"}
          note={health.guard_enabled ? "38 ms p50, decision 24" : "classifier disabled"}
          tone={health.guard_enabled ? "plain" : "warn"}
          tip={
            <>
              <strong>Two layers, orthogonal rather than stacked.</strong> The
              regex layer catches prompt injection, exfiltration attempts and
              residual secrets. Llama Guard catches harmful <em>content</em>.
              <br />
              <br />
              Measurement showed Llama Guard returns &ldquo;safe&rdquo; for a
              textbook injection, so neither substitutes for the other. Answers
              are screened on the way out too, not just prompts on the way in.
            </>
          }
        />
        <Cell
          label={health.frontier_available ? "level 2" : "air-gap"}
          value={health.frontier_available ? health.frontier_model : "OFFLINE"}
          note={
            health.frontier_available
              ? `${budgetValue} · can spend`
              : `${budgetValue} · escalation degrades`
          }
          tone={health.frontier_available ? "warn" : "good"}
          tip={
            health.frontier_available ? (
              <>
                <strong>Level 2 is armed.</strong> Credentials are present, so a
                question the local model cannot answer confidently may escalate
                to {health.frontier_model} and spend quota.
                <br />
                <br />
                The rolling window allows {budget?.messages_limit ?? "?"} messages;{" "}
                {budget?.messages_used ?? 0} used. When it is exhausted the gate
                vetoes every escalation and answers locally instead.
              </>
            ) : (
              <>
                <strong>No credentials, so nothing can leave this machine.</strong>{" "}
                Every escalation degrades to a local answer and records a{" "}
                <code>frontier_unavailable</code> flag in the ledger.
                <br />
                <br />
                This is a supported operating mode, not a fault &mdash; the
                budget counter stays at zero because there is nothing to spend.
              </>
            )
          }
        />
      </div>
    </div>
  );
}
