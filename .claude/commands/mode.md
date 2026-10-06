---
description: Switch operating mode (ba, architect, engineer, auditor, reset)
argument-hint: ba | architect | engineer | auditor | reset
---

Switch operating mode. Requested: `$1`

Aliases: `arch` → `architect`, `code` → `engineer`, `audit` → `auditor`.
`reset` → `engineer`, which is the default mode.

1. If the request is empty, state the current mode and list the valid names. Do not switch.
2. If it isn't a valid name or alias, say so, list the valid names, and stay in the current mode.
3. Otherwise, read `.claude/modes/<name>.md` with the Read tool.

After reading the file, adopt that persona: its focus, evaluation criteria,
boundaries, and output format. Keep it for every later turn until another
`/mode` or a shortcut (`/ba`, `/architect`, `/code`, `/audit`) switches it. The
project's `AGENTS.md` rules still apply in every mode.

Start every response with the mode's badge on its own line, exactly as written
in the file's `Badge:` line.

Reply to this command with only the badge and one sentence about what this
mode will do.
