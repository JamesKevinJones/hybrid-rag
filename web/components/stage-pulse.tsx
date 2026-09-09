import type { Stage } from "@/lib/types";

/**
 * The three pipeline phases, advanced by real `Stage` events from the server.
 *
 * There is no timer in here and no interpolation. A stage lights up when the
 * orchestrator says it started and not a millisecond before, which is the
 * whole reason `on_stage` exists on `execute()`. An indicator that animated on
 * its own schedule would be a fabricated measurement wearing a spinner - and
 * it would be wrong in exactly the interesting case, where the guard model is
 * slow and the UI claims to be reranking.
 *
 * In demo mode the same component is driven by the recorded timings, so it
 * still moves at the speed the pipeline actually ran.
 */

const STAGES: { id: Stage; label: string; detail: string }[] = [
  { id: "governance", label: "GOVERNANCE", detail: "scrub PII, screen for injection" },
  { id: "retrieval", label: "RETRIEVAL", detail: "embed, search, rerank" },
  { id: "generation", label: "GENERATION", detail: "resident 3B, or escalate" },
];

export function StagePulse({ current }: { current: Stage | null }) {
  const index = current ? STAGES.findIndex((stage) => stage.id === current) : -1;

  return (
    <ol className="flex flex-col gap-2 sm:flex-row sm:gap-3">
      {STAGES.map((stage, position) => {
        const done = index > position;
        const live = index === position;
        return (
          <li
            key={stage.id}
            className={`brut-sm flex flex-1 items-center gap-3 px-4 py-3 ${
              live ? "bg-acid" : ""
            }`}
            aria-current={live ? "step" : undefined}
          >
            <span
              aria-hidden
              className={`h-3.5 w-3.5 shrink-0 border-[3px] border-[var(--line)] ${
                done ? "bg-mint" : live ? "bg-ink stage-live" : ""
              }`}
            />
            <span className="min-w-0">
              <span className={`label block ${live ? "text-ink" : ""}`}>{stage.label}</span>
              <span
                className={`block truncate text-[0.8125rem] opacity-75 ${
                  live ? "text-ink" : ""
                }`}
              >
                {stage.detail}
              </span>
            </span>
          </li>
        );
      })}
    </ol>
  );
}
