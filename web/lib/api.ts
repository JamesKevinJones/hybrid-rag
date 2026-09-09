/**
 * Live client. Talks to `hybrid-rag serve` on the viewer's own machine.
 *
 * The SSE stream is read with `fetch` + a `ReadableStream` reader rather than
 * `EventSource`, for one reason that matters: `EventSource` cannot issue a
 * POST. The question has to go in a request body - putting a prompt in a query
 * string would write it into browser history and any proxy log, which is the
 * opposite of what a system that hashes prompts before storing them is for.
 */

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

/**
 * Where the API lives.
 *
 * Empty string means same-origin, which is the case when `hybrid-rag serve`
 * mounts the compiled UI at `/`. During `next dev` the UI is on :3000 and the
 * API on :8000, so the origin has to be spelled out - and that is exactly the
 * pair the server's CORS allowlist permits.
 */
export const API_BASE =
  process.env.NEXT_PUBLIC_HYBRID_RAG_API ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://127.0.0.1:8000"
    : "");

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`${path} responded ${response.status}`);
  }
  return (await response.json()) as T;
}


async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    // Always application/json. The server refuses anything else with a 415,
    // deliberately: a text/plain POST is a CORS "simple request" that skips
    // the preflight, which is how a foreign page could otherwise drive this.
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const parsed = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error((parsed as { error?: string }).error ?? `${path} responded ${response.status}`);
  }
  return parsed as T;
}

const LIVE_PROMPTS = [
  "How does the budget veto work?",
  "What happens when the frontier tier is unavailable?",
  "Why is the escalation threshold 0.35?",
  "How is the audit ledger tamper-evident?",
];

export function createLiveClient(): HybridRagClient {
  return {
    mode: "live",
    canMutate: true,
    models: () => getJson<ModelsInfo>("/api/models"),
    selectModel: (profile) => postJson<SelectResult>("/api/models/select", { profile }),
    setParams: (params) => postJson<GenerationParams>("/api/models/params", params),
    addProfile: (name, model) =>
      postJson<{ assessment: Assessment }>("/api/models/profiles", { name, model }),
    requirements: (model, numCtx) =>
      getJson<Assessment>(
        `/api/requirements?model=${encodeURIComponent(model)}&num_ctx=${numCtx}`,
      ),
    health: () => getJson<Health>("/api/health"),
    budget: () => getJson<Budget>("/api/budget"),
    ledger: (limit = 5) => getJson<Ledger>(`/api/ledger?limit=${limit}`),
    prompts: () => LIVE_PROMPTS,

    async ask(query, onStage, signal) {
      const response = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query }),
        signal,
      });
      if (!response.ok || !response.body) {
        throw new Error(`the server responded ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let result: ExecutionResult | null = null;

      // SSE frames are separated by a blank line. Chunk boundaries fall
      // wherever TCP decides, so frames are only parsed once one is complete.
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let split = buffer.indexOf("\n\n");
        while (split !== -1) {
          const frame = buffer.slice(0, split);
          buffer = buffer.slice(split + 2);
          split = buffer.indexOf("\n\n");

          let event = "message";
          let data = "";
          for (const line of frame.split("\n")) {
            if (line.startsWith("event: ")) event = line.slice(7).trim();
            else if (line.startsWith("data: ")) data += line.slice(6);
          }
          if (!data) continue;

          if (event === "stage") {
            onStage(JSON.parse(data).stage as Stage);
          } else if (event === "result") {
            result = JSON.parse(data) as ExecutionResult;
          }
        }
      }

      if (!result) {
        throw new Error("the stream ended without a result");
      }
      return result;
    },
  };
}
