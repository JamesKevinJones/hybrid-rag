"use client";

import { useState } from "react";

import type { Ledger } from "@/lib/types";

/**
 * The governance ledger, last few rows.
 *
 * Safe to render in a public demo for a structural reason, not a hopeful one:
 * `AuditRecord` stores `prompt_hash` and never prompt text, and guardrail
 * finding evidence is scrubbed before it is written. A test at the API
 * boundary asserts no raw prompt survives this endpoint.
 *
 * The chain-intact indicator is the point of the panel. Every record links to
 * the one before it, so an edited row and a deleted row are both detectable.
 * It is unkeyed SHA-256 - that catches accidental damage and casual edits, not
 * an attacker with write access who can recompute the chain, and the panel
 * says so rather than implying more.
 */
export function LedgerPanel({ ledger }: { ledger: Ledger | null }) {
  const [open, setOpen] = useState(false);

  if (!ledger) return null;

  return (
    <section className="brut mt-10">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="label flex w-full items-center justify-between gap-3 px-4 py-4 text-left"
      >
        <span>governance ledger</span>
        <span className="flex items-center gap-3">
          <span
            className={`px-2 py-1 ${ledger.intact ? "bg-mint text-ink" : "bg-coral text-ink"}`}
          >
            {ledger.intact ? "CHAIN INTACT" : `BROKEN AT LINE ${ledger.first_bad_line}`}
          </span>
          <span className="opacity-75">{ledger.records} RECORDS</span>
          <span aria-hidden className="font-mono">
            {open ? "-" : "+"}
          </span>
        </span>
      </button>

      {open && (
        <div className="border-t-[3px] border-[var(--line)]">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] border-collapse text-left">
              <thead>
                <tr className="border-b-[3px] border-[var(--line)]">
                  <th className="label px-4 py-2 font-normal">stage</th>
                  <th className="label px-4 py-2 font-normal">tier</th>
                  <th className="label px-4 py-2 font-normal">tokens</th>
                  <th className="label px-4 py-2 font-normal">conf</th>
                  <th className="label px-4 py-2 font-normal">record hash</th>
                </tr>
              </thead>
              <tbody className="font-mono text-[0.8125rem]">
                {ledger.rows.map((row, index) => (
                  <tr
                    key={row.record_hash ?? index}
                    className="border-b border-[var(--line)]/25 last:border-0"
                  >
                    <td className="px-4 py-2">{row.stage ?? "?"}</td>
                    <td className="px-4 py-2">
                      <span
                        className={`px-1.5 py-0.5 ${
                          row.tier === "opus" ? "bg-volt text-white" : "bg-mint text-ink"
                        }`}
                      >
                        {row.tier ?? "?"}
                      </span>
                    </td>
                    <td className="px-4 py-2 opacity-75">
                      {row.tokens_in ?? 0}+{row.tokens_out ?? 0}
                    </td>
                    <td className="px-4 py-2 opacity-75">
                      {typeof row.confidence === "number" ? row.confidence.toFixed(3) : "-"}
                    </td>
                    <td className="px-4 py-2 opacity-75">
                      {row.record_hash ? `${row.record_hash.slice(0, 12)}...` : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="border-t-[3px] border-[var(--line)] px-4 py-3.5 text-[0.8125rem] opacity-75">
            Prompts are stored as hashes, never as text. The chain is unkeyed
            SHA-256: it detects accidental damage and casual edits, not an
            attacker with write access who can recompute it.
          </p>
        </div>
      )}
    </section>
  );
}
