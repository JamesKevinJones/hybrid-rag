"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { Assessment, HybridRagClient, ModelsInfo } from "@/lib/types";

import { Tip } from "./tip";

/**
 * Model selection, generation parameters, and the "will it run here?" check.
 *
 * The capability check is the interesting part, and it is built to the same
 * rule as everything else in this project: weights and GPU capacity are
 * MEASURED (`ollama list`, `nvidia-smi`), the KV cache is an ESTIMATE scaled
 * from a single datapoint, and the panel says which is which. A capability
 * checker that presents a guess as a measurement is worse than no checker,
 * because it sounds authoritative.
 *
 * Read-only in demo mode - there is no server to mutate, and the panel says so
 * rather than offering controls that throw.
 */

const VERDICT_STYLE: Record<Assessment["verdict"], string> = {
  fits: "bg-mint text-ink",
  tight: "bg-acid text-ink",
  exceeds: "bg-coral text-ink",
  unknown: "bg-paper-dim text-ink",
};

const CTX_CHOICES = [2048, 4096, 8192, 16384, 32768];

function Row({ label, children, tip }: { label: string; children: React.ReactNode; tip?: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 border-b-[3px] border-[var(--line)] px-5 py-4 last:border-b-0">
      <span className="label opacity-75">
        {tip ? <Tip content={tip}>{label}</Tip> : label}
      </span>
      {children}
    </div>
  );
}

export function ModelPanel({
  client,
  onChanged,
}: {
  client: HybridRagClient | null;
  onChanged: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [info, setInfo] = useState<ModelsInfo | null>(null);
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [probe, setProbe] = useState("");
  const [probeCtx, setProbeCtx] = useState(4096);
  const [newName, setNewName] = useState("");
  const [newModel, setNewModel] = useState("");
  // Local, uncommitted temperature while the slider is being dragged. `null`
  // means "not dragging" - render from `info` instead. Without this the value
  // prop chases the server round-trip and the thumb visibly lags every tick.
  const [pendingTemp, setPendingTemp] = useState<number | null>(null);

  // Seed the context-length probe from the server ONCE. `load()` re-runs after
  // every mutation (temperature, profile switch, add-profile), and re-seeding
  // every time silently threw away whatever the user had picked in the "will
  // it run here?" select.
  const seededCtx = useRef(false);

  const load = useCallback(async () => {
    if (!client) return;
    try {
      const next = await client.models();
      setInfo(next);
      setProbe((current) => current || next.current_model);
      if (!seededCtx.current) {
        setProbeCtx(next.params.num_ctx);
        seededCtx.current = true;
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "could not read the model list");
    }
  }, [client]);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  const guard = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
      await load();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "that did not work");
    } finally {
      setBusy(false);
    }
  };

  const check = async (model: string, ctx: number) => {
    if (!client) return;
    setChecking(true);
    setError(null);
    try {
      setAssessment(await client.requirements(model, ctx));
    } catch (e) {
      setAssessment(null);
      setError(e instanceof Error ? e.message : "the check failed");
    } finally {
      setChecking(false);
    }
  };

  // `editable` alone does not let TypeScript narrow `client` through the
  // closures below, so every mutation still needed a `client!` assertion.
  // This is the single place that proves `client` is non-null for the rest
  // of the component - the panel simply doesn't render before mode resolves.
  if (!client) return null;

  const editable = client.canMutate;

  const commitTemp = () => {
    if (pendingTemp === null || pendingTemp === info?.params.temperature) {
      setPendingTemp(null);
      return;
    }
    void guard(() => client.setParams({ temperature: pendingTemp })).finally(() =>
      setPendingTemp(null),
    );
  };

  return (
    <section className="brut mt-10">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="label flex w-full items-center justify-between gap-3 px-4 py-4 text-left"
      >
        <span>model &amp; hardware</span>
        <span className="flex items-center gap-3">
          <span className="truncate font-mono text-[0.8125rem] normal-case opacity-75">
            {info?.current_model ?? ""}
          </span>
          <span aria-hidden className="font-mono">
            {open ? "-" : "+"}
          </span>
        </span>
      </button>

      {open && (
        <div className="border-t-[3px] border-[var(--line)]">
          {!editable && (
            <p className="border-b-[3px] border-[var(--line)] px-5 py-4 text-[0.9375rem] leading-relaxed">
              Read-only here. This is a recorded demo with no server behind it &mdash;
              run the console locally with <code>hybrid-rag serve</code> to switch
              models or check your own hardware.
            </p>
          )}

          {error && (
            <p className="border-b-[3px] border-[var(--line)] bg-coral px-5 py-3.5 text-[0.9375rem] text-ink">
              {error}
            </p>
          )}

          <Row
            label="generator profile"
            tip={
              <>
                Profiles are named model choices. Switching is a config change,
                not a download &mdash; the model has to be pulled already.
                <br />
                <br />
                The previous model stays in VRAM until Ollama evicts it, so both
                may be resident briefly after a switch.
              </>
            }
          >
            <div className="flex flex-wrap gap-2">
              {Object.entries(info?.profiles ?? {}).map(([name, model]) => (
                <button
                  key={name}
                  type="button"
                  disabled={!editable || busy || name === info?.current_profile}
                  onClick={() => void guard(() => client.selectModel(name))}
                  className={`brut-sm brut-press px-3 py-2 text-left font-mono text-[0.8125rem] ${
                    name === info?.current_profile ? "bg-mint text-ink" : ""
                  }`}
                >
                  <span className="label block">{name}</span>
                  <span className="opacity-75">{model}</span>
                </button>
              ))}
            </div>
          </Row>

          <Row
            label="temperature"
            tip={
              <>
                0 is deterministic. The audit ledger records the temperature
                beside the model digest so a decision can be replayed
                bit-for-bit, so raising this trades reproducibility for variety.
                <br />
                <br />
                The ledger will show whatever was actually used.
              </>
            }
          >
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={0}
                max={2}
                step={0.1}
                // NOT `|| busy`: `guard()` flips `busy` true the instant a
                // commit fires, and disabling the input mid-drag cancels the
                // pointer capture, snapping the thumb back before the release
                // even lands.
                disabled={!editable}
                value={pendingTemp ?? info?.params.temperature ?? 0}
                // Commit on release, not on every tick. `onChange` fires once
                // per pixel of drag - committing there meant one gesture from
                // 0.0 to 2.0 spawned twenty server round-trips (each of which
                // re-fetches models, health, budget and the ledger), and the
                // bound `value` chasing that round-trip is what made the
                // thumb visibly lag and snap back.
                onChange={(e) => setPendingTemp(Number(e.target.value))}
                onPointerUp={commitTemp}
                onBlur={commitTemp}
                className="h-2 flex-1 appearance-none border-[3px] border-[var(--line)] bg-[var(--panel)] accent-coral"
              />
              <span className="font-mono text-[0.9375rem] font-medium">
                {(pendingTemp ?? info?.params.temperature ?? 0).toFixed(1)}
              </span>
            </div>
          </Row>

          <Row
            label="context length"
            tip={
              <>
                The KV cache scales with this, and it is the fastest way out of
                the VRAM envelope. 4096 on a 3B Q4 costs roughly 400 MB.
                <br />
                <br />
                Past the envelope Ollama spills into system memory rather than
                failing, which on a single-channel machine is about ten times
                slower. Use the check below before raising it.
              </>
            }
          >
            <div className="flex flex-wrap gap-2">
              {CTX_CHOICES.map((ctx) => (
                <button
                  key={ctx}
                  type="button"
                  disabled={!editable || busy}
                  onClick={() => void guard(() => client.setParams({ num_ctx: ctx }))}
                  className={`brut-sm brut-press px-3 py-2 font-mono text-[0.8125rem] ${
                    ctx === info?.params.num_ctx ? "bg-mint text-ink" : ""
                  }`}
                >
                  {ctx.toLocaleString()}
                </button>
              ))}
            </div>
          </Row>

          {/* ---------------------------------------------- capability check */}
          <Row
            label="will it run here?"
            tip={
              <>
                Weights and GPU capacity are <strong>measured</strong> from{" "}
                <code>ollama list</code> and <code>nvidia-smi</code>. The KV
                cache is an <strong>estimate</strong>, scaled from one measured
                datapoint on this hardware, and is labelled as one.
                <br />
                <br />
                It counts the guard and embedder too, because the real question
                is whether the whole co-resident set fits, not the generator
                alone.
              </>
            }
          >
            <div className="flex flex-wrap items-center gap-2">
              <input
                value={probe}
                onChange={(e) => setProbe(e.target.value)}
                disabled={!editable}
                placeholder="model tag, e.g. qwen2.5:7b"
                className="brut-sm min-w-[14rem] flex-1 px-3 py-2 font-mono text-[0.8125rem] outline-none"
              />
              <select
                value={probeCtx}
                disabled={!editable}
                onChange={(e) => setProbeCtx(Number(e.target.value))}
                className="brut-sm px-3 py-2 font-mono text-[0.8125rem] outline-none"
              >
                {CTX_CHOICES.map((c) => (
                  <option key={c} value={c}>
                    {c.toLocaleString()} ctx
                  </option>
                ))}
              </select>
              <button
                type="button"
                disabled={!editable || checking || !probe.trim()}
                onClick={() => void check(probe.trim(), probeCtx)}
                className="brut-sm brut-press label bg-coral px-4 py-2.5 text-ink"
              >
                {checking ? "checking" : "check"}
              </button>
            </div>

            {assessment && (
              <div className="brut-flat mt-3 px-4 py-3.5">
                <p className="flex flex-wrap items-center gap-3">
                  <span className={`label px-2.5 py-1 ${VERDICT_STYLE[assessment.verdict]}`}>
                    {assessment.headline}
                  </span>
                  <span className="font-mono text-[0.8125rem] opacity-75">
                    {assessment.model}
                  </span>
                </p>
                <p className="mt-3 text-[0.9375rem] leading-relaxed">{assessment.detail}</p>

                {assessment.required_mb !== null && (
                  <dl className="mt-3.5 grid grid-cols-2 gap-x-5 gap-y-1.5 font-mono text-[0.8125rem]">
                    <dt className="opacity-75">weights (measured)</dt>
                    <dd>{assessment.weights_mb} MB</dd>
                    <dt className="opacity-75">kv cache (estimate)</dt>
                    <dd>~{assessment.kv_estimate_mb} MB</dd>
                    <dt className="opacity-75">guard + embedder</dt>
                    <dd>{assessment.companions_mb} MB</dd>
                    <dt className="opacity-75">total needed</dt>
                    <dd>{assessment.required_mb} MB</dd>
                    <dt className="opacity-75">usable VRAM</dt>
                    <dd>{assessment.usable_mb} MB</dd>
                  </dl>
                )}

                {assessment.details?.parameters && (
                  <p className="mt-3 font-mono text-[0.8125rem] opacity-75">
                    {assessment.details.architecture} &middot;{" "}
                    {assessment.details.parameters} &middot;{" "}
                    {assessment.details.quantization}
                  </p>
                )}

                {assessment.gpu && (
                  <p className="mt-2 font-mono text-[0.8125rem] opacity-75">
                    {assessment.gpu.name} &middot; {assessment.gpu.total_mb} MB total,{" "}
                    {assessment.gpu.free_mb} MB free now
                  </p>
                )}
              </div>
            )}
          </Row>

          {/* ------------------------------------------------- add a profile */}
          <Row
            label="add a model"
            tip={
              <>
                Registers an installed Ollama model as a new profile. It has to
                be pulled already &mdash; this does not download anything.
                <br />
                <br />
                A model larger than the envelope is <em>allowed</em>, not
                refused: whether a spill into system memory is worth it is your
                call. The fit is reported so the trade is never silent.
              </>
            }
          >
            <div className="flex flex-wrap items-center gap-2">
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                disabled={!editable}
                placeholder="profile name"
                className="brut-sm w-36 px-3 py-2 font-mono text-[0.8125rem] outline-none"
              />
              <input
                value={newModel}
                onChange={(e) => setNewModel(e.target.value)}
                disabled={!editable}
                placeholder="ollama model tag"
                list="installed-models"
                className="brut-sm min-w-[12rem] flex-1 px-3 py-2 font-mono text-[0.8125rem] outline-none"
              />
              <datalist id="installed-models">
                {(info?.installed ?? []).map((m) => (
                  <option key={m.name} value={m.name} />
                ))}
              </datalist>
              <button
                type="button"
                disabled={!editable || busy || !newName.trim() || !newModel.trim()}
                onClick={() =>
                  void guard(async () => {
                    const added = await client.addProfile(newName.trim(), newModel.trim());
                    setAssessment(added.assessment);
                    setNewName("");
                    setNewModel("");
                  })
                }
                className="brut-sm brut-press label px-4 py-2.5"
              >
                add
              </button>
            </div>

            {info?.installed && (
              <p className="mt-2 font-mono text-[0.75rem] opacity-75">
                installed:{" "}
                {info.installed.map((m) => `${m.name} (${m.size_mb} MB)`).join(" · ")}
              </p>
            )}
          </Row>
        </div>
      )}
    </section>
  );
}
