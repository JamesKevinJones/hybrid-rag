"use client";

import gsap from "gsap";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { createClient, resolveMode, type Mode } from "@/lib/client";
import { EASE, ENTER, STAGGER, useIsomorphicLayoutEffect } from "@/lib/motion";
import type { Budget, ExecutionResult, Health, Ledger, Stage } from "@/lib/types";

import { Hud } from "./hud";
import { LedgerPanel } from "./ledger-panel";
import { ModelPanel } from "./model-panel";
import { ResultBadge } from "./result-badge";
import { SiteHeader } from "./site-header";
import { StagePulse } from "./stage-pulse";

interface Turn {
  id: number;
  query: string;
  result: ExecutionResult | null;
  error: string | null;
}

/**
 * The console.
 *
 * Shaped like a chat client rather than a marketing page: one centred column,
 * a composer pinned to the bottom, and the conversation scrolling above it.
 * The thesis and the status panel are the EMPTY STATE - they are what fills
 * the column before there is anything to read, and they step aside the moment
 * there is. A hero that stays put above a growing transcript pushes the thing
 * you came for off the screen.
 *
 * Mode is resolved in an effect rather than during render. `resolveMode()`
 * reads `window.location`, which does not exist while the static export is
 * being prerendered.
 */
export function Console() {
  const [mode, setMode] = useState<Mode | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [budget, setBudget] = useState<Budget | null>(null);
  const [ledger, setLedger] = useState<Ledger | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [stage, setStage] = useState<Stage | null>(null);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [capturedAt, setCapturedAt] = useState<string | null>(null);
  const [prompts, setPrompts] = useState<string[]>([]);

  const nextId = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const emptyRef = useRef<HTMLDivElement>(null);
  // The in-flight `ask()` request, if any. Lets `goHome` actually cancel the
  // pipeline run rather than merely hiding its result - without this, leaving
  // mid-answer left the SSE stream running server-side (holding its budget
  // reservation open) and left `busy` stuck true with no visible request to
  // explain why the composer stayed disabled.
  const inflight = useRef<AbortController | null>(null);
  const client = useMemo(() => (mode ? createClient(mode) : null), [mode]);
  const hasTurns = turns.length > 0;

  useEffect(() => setMode(resolveMode()), []);

  const refresh = useCallback(async () => {
    if (!client) return;
    const [h, b, l] = await Promise.allSettled([
      client.health(),
      client.budget(),
      client.ledger(6),
    ]);
    if (h.status === "fulfilled") setHealth(h.value);
    if (b.status === "fulfilled") setBudget(b.value);
    if (l.status === "fulfilled") setLedger(l.value);
  }, [client]);

  useEffect(() => {
    if (!client) return;
    setCapturedAt(client.capturedAt ?? null);
    setPrompts(client.prompts());
    void refresh();
  }, [client, refresh]);

  /*
    Entrance.

    `fromTo` with EXPLICIT end values, not `from`. `gsap.from()` infers its
    destination from whatever the element looks like when the tween is built -
    and this effect re-runs when `mode` resolves from null, so the second pass
    read a mid-flight opacity as the target. The heading animated 0 -> 0.43 and
    stopped there: permanently half-faded, and measurably below AA. Spelling
    the destination out makes the tween idempotent however many times React
    runs the effect.

    `overwrite: "auto"` kills the earlier tween rather than racing it.

    Everything sits inside `matchMedia`, so a reduced-motion visitor registers
    no tweens at all - not tweens that run instantly. Nothing is hidden in the
    markup, so a bundle error leaves a readable page rather than a blank one.
  */
  useIsomorphicLayoutEffect(() => {
    if (!emptyRef.current || hasTurns) return;
    const context = gsap.context(() => {
      const mm = gsap.matchMedia();
      mm.add("(prefers-reduced-motion: no-preference)", () => {
        const rise = (
          selector: string,
          from: gsap.TweenVars,
          vars: gsap.TweenVars = {},
        ) =>
          gsap.fromTo(
            selector,
            { ...from, opacity: 0 },
            {
              y: 0,
              yPercent: 0,
              opacity: 1,
              duration: ENTER,
              ease: EASE,
              overwrite: "auto",
              ...vars,
            },
          );

        rise("[data-anim='line']", { yPercent: 110 }, { stagger: 0.09 });
        rise("[data-anim='sub']", { y: 18 }, { delay: 0.22 });
        rise("[data-anim='cell']", { y: 22 }, { duration: 0.45, stagger: STAGGER, delay: 0.3 });
        rise("[data-anim='chip']", { y: 14 }, { duration: 0.4, stagger: 0.04, delay: 0.45 });
      });
      return () => mm.revert();
    }, emptyRef);
    return () => context.revert();
  }, [mode, hasTurns]);

  /*
    Follow the conversation, the way a chat client does - but ONLY once there
    is a conversation. Without the guard this fires on mount and scrolls the
    empty state's heading up behind the header, so the first thing a visitor
    sees is a blank column with the status panel floating in it.
  */
  useEffect(() => {
    const node = scrollRef.current;
    if (!node || turns.length === 0) return;
    node.scrollTo({
      top: node.scrollHeight,
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
    });
  }, [turns, stage]);

  const ask = useCallback(
    async (query: string) => {
      if (!client || busy || !query.trim()) return;
      const id = nextId.current++;
      const controller = new AbortController();
      inflight.current = controller;
      setTurns((current) => [...current, { id, query, result: null, error: null }]);
      setDraft("");
      setBusy(true);
      setStage(null);
      try {
        const result = await client.ask(query, setStage, controller.signal);
        setTurns((current) =>
          current.map((turn) => (turn.id === id ? { ...turn, result } : turn)),
        );
        void refresh();
      } catch (error) {
        // A deliberate `goHome()` cancellation, not a failure - the turn is
        // already gone from `turns`, so there is nothing left to annotate.
        if (error instanceof DOMException && error.name === "AbortError") return;
        setTurns((current) =>
          current.map((turn) =>
            turn.id === id
              ? {
                  ...turn,
                  error:
                    error instanceof Error
                      ? `${error.message} - is \`hybrid-rag serve\` running?`
                      : "the request failed",
                }
              : turn,
          ),
        );
      } finally {
        if (inflight.current === controller) inflight.current = null;
        setBusy(false);
        setStage(null);
      }
    },
    [busy, client, refresh],
  );

  const goHome = useCallback(() => {
    inflight.current?.abort();
    inflight.current = null;
    setTurns([]);
    setDraft("");
    setStage(null);
    void refresh();
  }, [refresh]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <SiteHeader onHome={goHome} />
      <Hud health={health} budget={budget} />

      <div ref={scrollRef} data-scroll-region className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-4 sm:px-6">
          {hasTurns ? (
            <ol className="flex flex-col gap-10 py-10">
              {turns.map((turn) => (
                <TurnView key={turn.id} turn={turn} stage={stage} />
              ))}
            </ol>
          ) : (
            <EmptyState
              ref={emptyRef}
              mode={mode}
              capturedAt={capturedAt}
              prompts={prompts}
              busy={busy}
              onPick={ask}
            />
          )}

          <ModelPanel client={client} onChanged={() => void refresh()} />
          <LedgerPanel ledger={ledger} />
          <div className="h-8" />
        </div>
      </div>

      {/* The composer is chrome, not content: pinned, full-bleed rule, centred
          to the same measure as the transcript above it. */}
      <div className="sticky bottom-0 border-t-[3px] border-[var(--line)] bg-[var(--bg)]">
        <div className="mx-auto w-full max-w-3xl px-4 py-4 sm:px-6">
          <form
            id="composer"
            onSubmit={(event) => {
              event.preventDefault();
              void ask(draft);
            }}
          >
            <div className="flex items-end gap-3">
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  // Enter sends, Shift+Enter breaks the line.
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void ask(draft);
                  }
                }}
                rows={1}
                disabled={busy}
                aria-label="Prompt"
                placeholder={
                  mode === "demo" ? "Pick a recorded run below" : "Ask the local model"
                }
                className="brut max-h-40 min-h-[3.5rem] w-full resize-y px-4 py-3.5 font-mono text-base outline-none"
              />
              <button
                type="submit"
                disabled={busy || !draft.trim()}
                className="brut brut-press label shrink-0 bg-coral px-6 py-4 text-ink"
              >
                {busy ? "..." : "send"}
              </button>
            </div>

            <div className="mt-3 flex min-h-[1.5rem] items-center justify-between gap-4">
              <span className="label opacity-75">
                {busy ? (
                  <span aria-live="polite">{stage ?? "starting"}&hellip;</span>
                ) : (
                  <>enter to send &middot; shift+enter for a newline</>
                )}
              </span>
              {hasTurns && !busy && (
                <button
                  type="button"
                  onClick={() => setTurns([])}
                  className="label opacity-75 underline underline-offset-4 hover:opacity-100"
                >
                  clear
                </button>
              )}
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ turn */

function TurnView({ turn, stage }: { turn: Turn; stage: Stage | null }) {
  const ref = useRef<HTMLLIElement>(null);

  useIsomorphicLayoutEffect(() => {
    if (!ref.current) return;
    const context = gsap.context(() => {
      const mm = gsap.matchMedia();
      mm.add("(prefers-reduced-motion: no-preference)", () => {
        gsap.fromTo(
          ref.current,
          { y: 24, opacity: 0 },
          { y: 0, opacity: 1, duration: 0.4, ease: EASE, overwrite: "auto" },
        );
      });
      return () => mm.revert();
    }, ref);
    return () => context.revert();
  }, []);

  return (
    <li ref={ref}>
      {/* The question, right-aligned and boxed - the shape a chat client uses
          to say "this one was yours". */}
      <div className="flex justify-end">
        <p className="brut-sm max-w-[85%] bg-paper-dim px-4 py-3 font-mono text-base break-words text-ink dark:bg-[var(--panel)] dark:text-[var(--fg)]">
          {turn.query}
        </p>
      </div>

      {turn.result === null && turn.error === null && (
        <div className="mt-5">
          <StagePulse current={stage} />
        </div>
      )}

      {turn.error && (
        <p className="brut-sm mt-5 bg-coral px-4 py-3.5 text-base text-ink">{turn.error}</p>
      )}

      {turn.result && (
        <div className="mt-5">
          {turn.result.answer ? (
            <div className="brut px-5 py-5 sm:px-6">
              <p className="whitespace-pre-wrap text-[1.0625rem] leading-[1.75]">
                {turn.result.answer}
              </p>
            </div>
          ) : (
            <div className="brut overflow-hidden">
              <div className="hazard h-3.5" aria-hidden />
              <div className="px-5 py-5 sm:px-6">
                <p className="label mb-2.5">refused</p>
                <p className="text-base leading-relaxed">{turn.result.reason}</p>
              </div>
            </div>
          )}
          {turn.result.degraded_reason && turn.result.answer && (
            <p className="mt-2.5 text-sm opacity-75">{turn.result.degraded_reason}</p>
          )}
          <ResultBadge result={turn.result} />
        </div>
      )}
    </li>
  );
}

/* ------------------------------------------------------------ empty state */

function EmptyState({
  ref,
  mode,
  capturedAt,
  prompts,
  busy,
  onPick,
}: {
  ref: React.Ref<HTMLDivElement>;
  mode: Mode | null;
  capturedAt: string | null;
  prompts: string[];
  busy: boolean;
  onPick: (prompt: string) => void;
}) {
  return (
    <div ref={ref} className="flex min-h-full flex-col justify-center py-12 sm:py-16">
      <h1 className="text-[clamp(1.9rem,5.4vw,3.4rem)]">
        {/* Each line is its own overflow-clipped row so the stagger can slide
            them up from behind their own edge rather than fading in place. */}
        <span className="block overflow-hidden py-[0.06em]">
          <span data-anim="line" className="block">
            Cloud tokens are
          </span>
        </span>
        <span className="block overflow-hidden py-[0.06em]">
          <span data-anim="line" className="block">
            a <span className="mark">budgeted</span> resource.
          </span>
        </span>
      </h1>

      <p data-anim="sub" className="mt-6 max-w-xl text-[1.0625rem] leading-[1.7]">
        Level 0 runs on a 6 GB GPU. Level 2 is Claude Opus 5 and costs quota. A
        three-tier gate decides which one answers, and every call lands in a
        hash-chained ledger &mdash; including the ones that stayed local.
      </p>

      {mode === "demo" && (
        <section data-anim="sub" className="brut mt-8 overflow-hidden">
          <div className="hazard hazard-live h-3.5" aria-hidden />
          <div className="px-5 py-5">
            <p className="label mb-2.5">what this page will not do</p>
            <p className="text-base leading-relaxed">
              This is a <strong>recorded</strong> demo. There is no model behind it
              &mdash; hybrid-rag runs a 3B on a local GPU with an on-disk index, and a
              web host has neither. Every number below came from a real run
              {capturedAt ? ` on ${capturedAt.slice(0, 10)}` : ""}, replayed at the
              speed it actually took. Anything unrecorded is refused rather than
              matched to the nearest answer.
            </p>
          </div>
        </section>
      )}

      {prompts.length > 0 && (
        <section className="mt-8">
          <p className="label mb-3.5 opacity-75">
            {mode === "demo" ? "recorded runs" : "try one"}
          </p>
          <div className="flex flex-col gap-2.5">
            {prompts.map((prompt) => (
              <button
                key={prompt}
                data-anim="chip"
                type="button"
                disabled={busy}
                onClick={() => onPick(prompt)}
                className="brut-sm brut-press w-full px-4 py-3.5 text-left font-mono text-[0.9375rem]"
              >
                {prompt.length > 90 ? `${prompt.slice(0, 90)}…` : prompt}
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
