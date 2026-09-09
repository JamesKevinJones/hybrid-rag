/**
 * Demo client. Replays runs that actually happened.
 *
 * Every byte in `fixtures/bundle.json` was produced by
 * `scripts/capture_web_fixtures.py` driving the real pipeline on real
 * hardware - real confidences, real latencies, real ledger hashes. Nothing in
 * here is authored by hand, which is what lets the deployed page stay inside
 * the project's own rule that a displayed number is measured, cited or absent.
 *
 * The one thing it will not do is answer an unrecorded question. A demo that
 * quietly fuzzy-matched a stranger's prompt onto the nearest recording would
 * be presenting a fabricated answer as a live one, which is the exact failure
 * this whole system is built to make visible.
 */

import bundle from "../fixtures/bundle.json";
import type {
  Assessment,
  Budget,
  ExecutionResult,
  GenerationParams,
  Health,
  HybridRagClient,
  Ledger,
  ModelsInfo,
  SelectResult,
  Stage,
} from "./types";

interface StageEvent {
  stage: Stage;
  at_ms: number;
}

interface Fixture {
  label: string;
  prompt: string;
  stages: StageEvent[];
  result: ExecutionResult;
}

interface Bundle {
  captured_at: string;
  health: Health;
  budget: Budget;
  ledger: Ledger;
  fixtures: Fixture[];
}

const data = bundle as unknown as Bundle;

const normalise = (text: string) => text.trim().toLowerCase().replace(/\s+/g, " ");

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Shown when someone asks something the capture never recorded. */
function unrecorded(query: string): ExecutionResult {
  return {
    trace_id: "demo-unrecorded",
    outcome: "blocked",
    answer: "",
    reason:
      `This is a recorded demo, and "${query.slice(0, 80)}" is not one of the ` +
      "captured runs. Pick a prompt below, or run the console locally against " +
      "your own models for live answers.",
    audit_hash: null,
    retrieval_confidence: 0,
    routing_stage: null,
    degraded_reason: null,
    frontier_unavailable: false,
    pii_status: "not_evaluated",
    pii_entities: 0,
    local_tokens_in: 0,
    local_tokens_out: 0,
    cloud_tokens_in: 0,
    cloud_tokens_out: 0,
    latency_ms: 0,
    generation_ms: 0,
    sources: [],
    quarantined_chunks: 0,
    distilled: false,
    spent_cloud_tokens: false,
    token_source: "none",
  };
}

export function createDemoClient(): HybridRagClient {
  const noServer = () => {
    throw new Error(
      "This is a recorded demo with no server behind it. Run the console " +
        "locally with `hybrid-rag serve` to change models or check what your " +
        "machine can run.",
    );
  };

  return {
    mode: "demo",
    capturedAt: data.captured_at,
    canMutate: false,
    // Readable from the capture; the mutations have nothing to talk to.
    models: async (): Promise<ModelsInfo> => ({
      current_profile: data.health.generator_profile,
      current_model: data.health.generator,
      profiles: { [data.health.generator_profile]: data.health.generator },
      params: { temperature: 0, num_ctx: 4096, keep_alive: -1, seed: 1337 },
      installed: null,
      gpu: null,
      companions_mb: 0,
    }),
    selectModel: async (): Promise<SelectResult> => noServer(),
    setParams: async (): Promise<GenerationParams> => noServer(),
    addProfile: async (): Promise<{ assessment: Assessment }> => noServer(),
    requirements: async (): Promise<Assessment> => noServer(),
    health: async () => data.health,
    budget: async () => data.budget,
    ledger: async (limit = 5) => ({
      ...data.ledger,
      rows: data.ledger.rows.slice(-limit),
    }),
    prompts: () => data.fixtures.map((fixture) => fixture.prompt),

    async ask(query, onStage, signal) {
      const match = data.fixtures.find(
        (fixture) => normalise(fixture.prompt) === normalise(query),
      );
      if (!match) {
        await sleep(120);
        return unrecorded(query);
      }

      // Replay the recorded timings so the stage indicator moves at the speed
      // the pipeline actually ran, rather than at a speed that flatters it.
      let elapsed = 0;
      for (const event of match.stages) {
        if (signal?.aborted) throw new DOMException("aborted", "AbortError");
        await sleep(Math.max(0, Math.min(event.at_ms - elapsed, 2500)));
        elapsed = event.at_ms;
        onStage(event.stage);
      }
      await sleep(Math.max(0, Math.min(match.result.latency_ms - elapsed, 2500)));
      return match.result;
    },
  };
}
