/**
 * Mirrors the Python models exactly. These names are not a convenience layer -
 * they are the wire contract with `src/hybrid_rag/web/app.py`, and every field
 * here exists on the server side.
 *
 * Rule 14 of AGENTS.md applies to this file: if a display needs a number the
 * pipeline does not produce, the fix is to add it to `ExecutionResult`, not to
 * derive a plausible one here.
 */

/** `Outcome` in agents/orchestrator.py */
export type Outcome = "local" | "local_degraded" | "opus" | "blocked" | "failed";

/** `Stage` in agents/orchestrator.py. Three, and no token-level streaming. */
export type Stage = "governance" | "retrieval" | "generation";

/** `PIIStatus` in governance/audit.py */
export type PiiStatus = "not_evaluated" | "clean" | "scrubbed" | "blocked";

/** `TokenSource` in governance/audit.py */
export type TokenSource = "local_ollama" | "cloud_anthropic" | "cache" | "none";

/** `SourceAttribution` in governance/audit.py */
export interface SourceAttribution {
  doc_id: string;
  chunk_id: string;
  uri: string | null;
  score: number;
  rank: number;
}

/** `ExecutionResult` in agents/orchestrator.py, including its computed fields. */
export interface ExecutionResult {
  trace_id: string;
  outcome: Outcome;
  answer: string;
  reason: string;
  audit_hash: string | null;
  retrieval_confidence: number;
  routing_stage: string | null;
  degraded_reason: string | null;
  frontier_unavailable: boolean;
  pii_status: PiiStatus;
  pii_entities: number;
  local_tokens_in: number;
  local_tokens_out: number;
  cloud_tokens_in: number;
  cloud_tokens_out: number;
  latency_ms: number;
  generation_ms: number;
  sources: SourceAttribution[];
  quarantined_chunks: number;
  distilled: boolean;
  /** `@computed_field` - would have been dropped by a plain `@property`. */
  spent_cloud_tokens: boolean;
  token_source: TokenSource;
}

export interface ResidentModel {
  name: string;
  size: string;
  processor: string;
}

/** `GET /api/health` */
export interface Health {
  ollama_up: boolean;
  ollama_url: string;
  generator: string;
  generator_profile: string;
  generator_resident: boolean;
  residency: string;
  /** `null` means "could not ask", which is NOT the same as "nothing loaded". */
  resident_models: ResidentModel[] | null;
  embed_model: string;
  embed_dim: number;
  rerank_model: string;
  rerank_enabled: boolean;
  guard_model: string;
  guard_enabled: boolean;
  indexed_chunks: number;
  escalation_threshold: number;
  frontier_available: boolean;
  frontier_model: string;
}

/** `GET /api/budget` - `BudgetSnapshot` plus two derived fields. */
export interface Budget {
  window_hours: number;
  messages_used: number;
  messages_limit: number;
  messages_remaining: number;
  tokens_used: number;
  tokens_limit: number;
  tokens_remaining: number;
  weekly_messages_used: number;
  weekly_messages_limit: number;
  /** `@computed_field` on `BudgetSnapshot`. */
  weekly_messages_remaining: number;
  open_reservations: number;
  seconds_to_window_reset: number;
  /** `@computed_field` on `BudgetSnapshot` - covers all three caps `reserve()` enforces. */
  exhausted: boolean;
}

/** One row of `AuditRecord`. Carries `prompt_hash`, never prompt text. */
export interface LedgerRow {
  trace_id?: string;
  stage?: string;
  tier?: string;
  tokens_in?: number;
  tokens_out?: number;
  confidence?: number | null;
  record_hash?: string | null;
  compliance_flags?: string[];
  timestamp?: number;
}

/** `GET /api/ledger` */
export interface Ledger {
  path: string;
  intact: boolean;
  first_bad_line: number;
  records: number;
  rows: LedgerRow[];
}

/**
 * `POST /api/models/select` in web/app.py - deliberately NOT `ModelsInfo`.
 * The route never re-fetches installed models or the GPU, so it cannot
 * return the fields `ModelsInfo` promises; claiming that type let a caller
 * read `installed`/`gpu`/`companions_mb` as `undefined` with no type error.
 */
export interface SelectResult {
  current_profile: string;
  current_model: string;
  note: string;
}

/** What both clients implement. The console never knows which one it has. */
export interface HybridRagClient {
  readonly mode: "live" | "demo";
  /** Present only in demo mode: when the fixtures were recorded. */
  readonly capturedAt?: string;
  health(): Promise<Health>;
  budget(): Promise<Budget>;
  ledger(limit?: number): Promise<Ledger>;
  /** Suggested prompts. Demo mode can only answer the ones it recorded. */
  prompts(): string[];
  ask(
    query: string,
    onStage: (stage: Stage) => void,
    signal?: AbortSignal,
  ): Promise<ExecutionResult>;

  /** False in demo mode: there is no server to mutate. */
  readonly canMutate: boolean;
  models(): Promise<ModelsInfo>;
  selectModel(profile: string): Promise<SelectResult>;
  setParams(params: Partial<Omit<GenerationParams, "seed">>): Promise<GenerationParams>;
  addProfile(name: string, model: string): Promise<{ assessment: Assessment }>;
  requirements(model: string, numCtx: number): Promise<Assessment>;
}

/** `GpuInfo` in hardware.py. `null` anywhere means "could not measure". */
export interface GpuInfo {
  name: string;
  total_mb: number;
  free_mb: number;
  usable_mb: number;
  driver: string;
}

export interface InstalledModel {
  name: string;
  digest: string;
  size_mb: number;
}

/** `GET /api/models` */
export interface ModelsInfo {
  current_profile: string;
  current_model: string;
  profiles: Record<string, string>;
  params: GenerationParams;
  /** `null` means Ollama could not be asked - NOT an empty install. */
  installed: InstalledModel[] | null;
  gpu: GpuInfo | null;
  companions_mb: number;
}

export interface GenerationParams {
  temperature: number;
  num_ctx: number;
  keep_alive: number | string;
  seed: number;
}

export type FitVerdict = "fits" | "tight" | "exceeds" | "unknown";

/**
 * `Assessment` in hardware.py.
 *
 * `weights_mb`, `usable_mb` and the GPU figures are MEASURED.
 * `kv_estimate_mb` is an estimate scaled from a single datapoint, which is why
 * it is named that way and why `kv_is_estimate` rides along.
 */
export interface Assessment {
  model: string;
  verdict: FitVerdict;
  headline: string;
  detail: string;
  installed: boolean;
  weights_mb: number | null;
  kv_estimate_mb: number | null;
  companions_mb: number;
  required_mb: number | null;
  usable_mb: number | null;
  gpu: GpuInfo | null;
  details: {
    architecture: string | null;
    parameters: string | null;
    quantization: string | null;
    context_length: number | null;
  } | null;
  num_ctx?: number;
  kv_is_estimate?: boolean;
}
