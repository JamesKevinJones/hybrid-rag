# State

_Updated 2026-09-06. See 'Where to pick up' at the end._

## Where we stopped

Modules 1 through 6 are built, calibrated against external data, and running.
331 tests, ruff clean, mypy strict clean, tree clean.

The system is **air-gapped by default and complete offline**. Level 2 is
optional wiring, not a dependency.

## Shipped 2026-09-06 (Module 6 + MCP hardening)

- **Terminal UI** at `src/hybrid_rag/ui/tui.py`. Bare `uv run hybrid-rag` opens
  it; `start_rag.bat` launches it from Explorer. Chat loop over the same
  `AgentOrchestrator.execute` path `ask` uses, so it adds no capability and
  bypasses no gate. `/help /doctor /budget /audit /model /clear /exit`.
- **Every displayed value is live, cited, or absent** (decision 29). Residency
  from `ollama ps` at render time, budget from the ledger, confidence and
  latency from the result object, ledger hash from the row that turn wrote.
  The two benchmark figures on screen carry the decision that recorded them.
- **Three orchestrator additions the UI needed**, added to the pipeline rather
  than faked in the renderer: a `Stage` callback so the spinner is event-driven,
  `generation_ms` so throughput is not diluted by retrieval, and `audit_hash` /
  `pii_status` / `pii_entities` on `ExecutionResult`.
- **`Container.switch_generator()`** re-points the generator AND the tier-3
  voter, which held a snapshot of the name. Restricted to declared profiles:
  the co-residency envelope was measured against ~3B Q4 weights and a 7B typed
  at a prompt would spill to shared memory rather than fail.
- **MCP launch command corrected** (decision 30). `-m src.hybrid_rag...` starts
  from the repo root and nowhere else, and imports the package twice when it
  does. Now `-m hybrid_rag.mcp_servers.server` via the venv interpreter.
- **`scripts/verify_mcp_clients.py`** - 14 checks, real stdio JSON-RPC against
  a subprocess started from a foreign working directory. Its own tests assert
  it rejects the three regressions it exists to catch.
- **Antigravity tool registry** in `instructions.md` with schemas dumped from
  the running server, not written by hand.
- Two defects found by running it: the `ollama ps` parser dropped half the
  PROCESSOR column ("100%" instead of "100% GPU"), and FlashRank's
  `basicConfig(INFO)` was turning on httpx request logging over the whole UI.
  Both fixed; the second by claiming the root logger at the entry point.

## Shipped 2026-09-05

- **Budget caps set to deliberate under-estimates**: 25 messages / 400k tokens
  per 5h window, 250 weekly. Not the published Pro limits and not trying to be.
  Guessing low costs a few answers; guessing high costs a lockout mid-session.
- **`.env` key resolution**, which had a real gap: `env_file` maps only
  PREFIXED names onto settings fields and never touches `os.environ`, so an
  ordinary `ANTHROPIC_API_KEY=` line was invisible to the one code path reading
  it. `get_settings()` now loads `.env` first, `override=False`.
- **Air-gap guarantee enforced.** Absent key, malformed key, dead DNS, refused
  handshake, 429, timeout - all arrive as `OpusUnavailable`, all degrade to a
  local answer with zero tokens and the reservation released. New
  `frontier_unavailable` field and compliance flag, because "unreachable",
  "not needed" and "refused" are three different facts.
- **Generator profiles**: `general` (llama3.2:3b) and `code`
  (qwen2.5-coder:3b), plus an explicit override. Unknown profile raises.
- **Threshold recalibrated 0.78 -> 0.35** against an external PEP corpus.
- **Llama Guard chained in** as a content-safety layer, orthogonal to the regex
  injection layer, 38 ms p50, INGRESS/EGRESS only.
- **Presidio verified** for the first time and kept opt-in on measured grounds.
- **AST validation** in the eval harness (`python_syntax` tag) and the
  promotion latency gate moved to **p50 <= 1200 ms** per spec, with p95 still
  reported.
- CI backstop: reverted a fix of mine that produced a green check reviewing
  nothing. See decision 26.

## Measured facts

```
VRAM, all 100% GPU:
  generator llama3.2:3b   2.6 GB     code profile qwen2.5-coder:3b  2.2 GB
  llama-guard3:1b         1.5 GB     llama-guard3:1b                1.5 GB
  all-minilm:l6-v2        0.03 GB    all-minilm:l6-v2               0.03 GB
                          4.1 GB                                    3.7 GB
                                     inside ~5.2 GB usable

Routing separation, external PEP corpus, 665 chunks, 18 labelled queries:
  answerable    n=10  min 0.4984  median 0.9413
  unanswerable  n= 8  max 0.0028  median 0.0000

Guardrail scope:
  regex        catches injection      llama-guard returns `safe` for injection
  llama-guard  catches S1-S13 content regex does not model content harm
```

## Calibration verification (re-run at closeout)

`uv run python scripts/calibrate_thresholds.py`, against the 665-chunk external
PEP corpus, 18 labelled queries:

```
answerable    n=10  min 0.4984  median 0.9413
unanswerable  n= 8  max 0.0028  median 0.0000
clean separation: (0.0028, 0.4984) - width 0.4956

  OK 0.250  false escalations 0/10   false locals 0/8
  OK 0.350  false escalations 0/10   false locals 0/8   <- locked
  OK 0.450  false escalations 0/10   false locals 0/8
     0.600  false escalations 2/10   false locals 0/8
     0.780  false escalations 3/10   false locals 0/8
```

**0.35 holds.** Reproducible from a clean checkout via
`fetch_calibration_corpus.py` then `calibrate_thresholds.py`.

## Air-gap closeout

Verified with no key present, stdin closed:

```
uv run hybrid-rag ask "Compare every retrieval strategy across all sources"
  -> outcome local_degraded, "Level 2 unavailable (no credentials)"
  -> tokens: local 1090+426  cloud 0+0
  -> exit code 0, no prompt, no error
```

`scripts/setup_env.py` creates a gitignored `.env` from the example, refuses to
write if `git check-ignore` says the file would be trackable, never overwrites
an existing one, and never prints a key.

`scripts/install-wslconfig.ps1 -Check` reports readiness without touching
anything. It found that **this host already has a `.wslconfig`** with different
caps - 10 GB memory / 6 processors against the template's 6 GB / 4 - so
installing would overwrite a deliberate choice. Left for the user to decide.

## IDE integration

- `.vscode/` - tasks (MCP server, doctor, ollama ps, eval tests, full gate,
  recalibration, env setup), interpreter pinned to `.venv/Scripts/python.exe`,
  and `mcp.json` launching `${workspaceFolder}/.venv/Scripts/python.exe -m
  hybrid_rag.mcp_servers.server`. JSONC, and the comments explaining why that
  module path are load-bearing - see decision 30.
- `.antigravity/instructions.md` - agent persona plus a full tool registry:
  launch command, exact input schemas (dumped from the running server) and
  return shapes for all three tools. Retrieval-first, egress only through the
  routing gate, structured artifacts per turn.
- `.antigravity/antigravity.json` - **schema unverified.** agy 1.1.12 has no
  `mcp` subcommand and no discoverable project-config file. The values inside
  are correct; the key names are a guess, and the file says so. The
  `before_apply_diff` hook COMMAND is verified - `cleared` / exit 0 on an
  ordinary diff, `live credential material detected` / exit 1 on one carrying a
  key - but nothing is known to invoke it, so `instructions.md` also tells the
  agent to run it by hand.
- **Cursor skipped - not installed.** Only a leftover `%APPDATA%\Cursor` cache
  directory; no executable anywhere on this machine.

## Where to pick up

The project is at a deliberate stopping point. Nothing is half-written, the
tree is clean, and `main` is pushed.

**Operating mode: OFFLINE, by decision 28.** No `ANTHROPIC_API_KEY` is
configured and that is the accepted state. Every escalation degrades to a local
answer with zero cloud tokens and a `frontier_unavailable` ledger flag. Do not
treat this as a misconfiguration to fix.

**MCP tool names are frozen** (decision 28): server `hybrid-rag`, tools
`local_rag_search`, `check_budget_status`, `verify_audit_ledger`. The
`LocalGovEngine` / `search_local_knowledge` names were rejected, not forgotten.

### If you want to keep building offline

Everything here needs no key and no network:

1. **Widen the calibration set.** 18 queries over 8 PEPs moved the threshold
   off a guessed value; it is not enough to claim a routing-percentage target.
   Add queries and PEPs, re-run `scripts/calibrate_thresholds.py`.
2. **Anchor the audit chain off-machine.** It is unkeyed SHA-256 today - it
   detects accidents and casual edits, not an attacker who recomputes it.
   `governance/audit.py` states that threat model at the top.
3. **Exercise the `code` generator profile.** `/model code` in the TUI switches
   to `qwen2.5-coder:3b` and the swap is tested, but no evaluation has compared
   its answers against the `general` profile on the same cases.
4. **Build an eval case set** against `data/calibration`. `evaluate` works with
   a local candidate on both sides; only the Opus baseline needs a key.

### If you ever want Level 2

```
uv run python scripts/setup_env.py
# uncomment ANTHROPIC_API_KEY in the generated .env, paste a key
uv run hybrid-rag ask "<a genuinely hard question>"
```

Confirm the ledger shows `tier: opus` with real token counts. That single call
unblocks the distillation dataset, the Opus baseline, `train_colab.py` and the
promotion gate - all four are built and waiting on it.

### Verify anything still works

```
uv run ruff check src tests && uv run mypy && uv run pytest
```

331 tests, no network, no Ollama required. Add `scripts` to the ruff paths if
you touched anything there. `docs/VERIFY.md` has the rest, including the
by-eye checks for the TUI and `scripts/verify_mcp_clients.py` for MCP.

## Known gaps (all deliberate)

- **Zero cloud tokens have ever been spent.** By decision 28, this is the
  intended state. The live client is built, typed and hardened but unused.
- `train_colab.py`, the Opus baseline, `evaluate` and `promote` are downstream
  of a first live call. Built, unit-tested, never run against a real model.
- Calibration is 18 queries over 8 PEPs. Enough to move a threshold off a
  guess; not enough to quote a routing percentage.
- The audit chain is unkeyed SHA-256; off-machine anchoring is unbuilt.
- `scripts/install-wslconfig.ps1` is NOT run. This host already has a
  `.wslconfig` at 10 GB / 6 processors against the template's 6 GB / 4, so
  installing would overwrite a deliberate choice.
- CI security review fires only on `pull_request`. Direct pushes to main get a
  `note-unreviewed` annotation and nothing else.
- `.antigravity/antigravity.json` has an unverified schema; the file says so.
- Cursor config absent - Cursor is not installed on this machine.
- The TUI needs a real Windows console. MinTTY (Git Bash) and any redirected or
  piped shell cannot host it; both cases print an actionable message rather than
  a Win32 traceback, but neither can run the UI.
- Prompt history is in-memory unless `--history-file` is passed. A prompt can
  hold exactly the data the scrubber exists to keep off disk, so persisting it
  is opt-in.
