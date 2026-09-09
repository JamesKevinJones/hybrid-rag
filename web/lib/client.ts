/**
 * Which client the console gets, decided at runtime rather than at build time.
 *
 * The build-time env var alone is not enough. One artifact is deployed to
 * Vercel *and* mounted by `hybrid-rag serve`, and those two need opposite
 * behaviour from identical bytes. Baking the mode in would force two builds
 * and guarantee that one day the wrong one ships.
 *
 * So the rule is: an explicit `demo` build stays demo everywhere, and
 * otherwise the page asks where it is running. On localhost there is plausibly
 * a server to talk to; anywhere else there certainly is not.
 *
 * The fallback is `demo` in every ambiguous case, including during SSR, where
 * `window` does not exist. That direction matters: a demo badge on a live
 * console is a cosmetic error, while a live console that cannot reach an API
 * is a broken page full of failed requests.
 */

import { createLiveClient } from "./api";
import { createDemoClient } from "./fixtures";
import type { HybridRagClient } from "./types";

export type Mode = "live" | "demo";

export function resolveMode(): Mode {
  if (process.env.NEXT_PUBLIC_HYBRID_RAG_MODE === "demo") return "demo";
  if (process.env.NEXT_PUBLIC_HYBRID_RAG_MODE === "live") return "live";

  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    const isLocal = host === "localhost" || host === "127.0.0.1" || host === "[::1]";
    return isLocal ? "live" : "demo";
  }
  return "demo";
}

export function createClient(mode: Mode = resolveMode()): HybridRagClient {
  return mode === "live" ? createLiveClient() : createDemoClient();
}
