# hybrid-rag console

The web UI. Neo-brutalist, same design language as
[portfolio-website](https://github.com/JamesKevinJones/portfolio-website) and
StarMatch: 3px ink borders, hard offset shadows, no blur and no gradients on
chrome. Coral is this project's accent because it is the palette colour nearest
the terminal UI's terracotta, so the TUI and the console read as one product.

## Two modes, one build

The mode is resolved at **runtime**, not baked in at build time, so a single
artifact serves both hosts:

| where | mode | data |
|---|---|---|
| `localhost` / `127.0.0.1` | **live** | your real pipeline via `hybrid-rag serve` |
| anywhere else | **demo** | `fixtures/bundle.json`, captured from real runs |

`NEXT_PUBLIC_HYBRID_RAG_MODE=demo` (or `live`) forces one, overriding detection.

The demo is a **recording**, not a mock. Every confidence, latency, token count
and ledger hash in `fixtures/bundle.json` came from
`scripts/capture_web_fixtures.py` driving the real orchestrator on real
hardware. It replays the recorded stage timings, and refuses unrecorded prompts
rather than matching them to the nearest recording - see decision 33.

## Running it

**Live, one command.** Build once, then let the Python server host it:

```bash
npm install
npm run build
cd .. && uv run hybrid-rag serve
```

That serves the UI and the API together on `http://127.0.0.1:8000`.

**Developing.** Two processes, with hot reload:

```bash
cd .. && uv run hybrid-rag serve    # API on :8000
npm run dev                        # UI on :3000, calls :8000
```

`:3000` is in the server's CORS allowlist for exactly this.

> Close the TUI first. Embedded Qdrant takes an exclusive lock on `data/index`,
> so `serve` and the TUI cannot both run. The server says so plainly if the
> lock is taken.

## Gates

```bash
npm run build      # the real gate - type errors fail it
npx tsc --noEmit   # types only
```

There is no test runner here, matching the portfolio. Contrast changes are
**measured** in the browser in both themes, not eyeballed.

## Rules carried over

- `@custom-variant dark (&:is(.dark *));` must stay at the top of
  `globals.css`. Without it the `dark:` utilities follow the OS while the
  toggle moves `.dark`, and a dark-set OS renders light mode at 1.14:1.
- **Coral carries ink text, never white.** White on coral is 3.05:1 and fails
  AA; ink on coral is 6.46:1. Only volt carries white, at 4.93:1.
- **Hazard striping marks refusal and nothing else** - the demo disclosure and
  a refused turn. The moment it becomes texture it stops reading as a warning.
- Theme lives in the DOM, set by a pre-paint script. `suppressHydrationWarning`
  on `<html>` is deliberate.
- Tailwind v4 with no config file. Tokens live in `globals.css` under `@theme`.
- lucide-react v1 dropped brand icons; do not `import { Github }` from it.

## Deploying

New Vercel project, root directory `web/`. **Do not point it at the
`portfolio-website` project** - that URL is on a resume and in every repo
README.

No env var is needed: runtime detection makes any non-localhost host demo mode.

Deploying is not pushing. After a push, fetch the live URL and confirm the
change is actually in the response.
