"use client";

import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

import { resolveMode, type Mode } from "@/lib/client";

/**
 * Theme is mutated directly on the document, not held in React state.
 *
 * The pre-paint script in `layout.tsx` is the only thing that sets the initial
 * value, and this button moves the same two attributes. CSS picks the icon, so
 * there is no state to get out of sync and no wrong glyph on first paint.
 */
function toggleTheme() {
  const root = document.documentElement;
  const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  root.classList.toggle("dark", next === "dark");
  try {
    localStorage.setItem("hybrid-rag-theme", next);
  } catch {
    /* Storage blocked. The toggle still works for this page view. */
  }
}

export function SiteHeader({ onHome }: { onHome?: () => void }) {
  const [mode, setMode] = useState<Mode | null>(null);
  useEffect(() => setMode(resolveMode()), []);

  return (
    <header className="shrink-0 border-b-[3px] border-[var(--line)] bg-[var(--bg)]">
      <div className="mx-auto flex w-full max-w-3xl flex-wrap items-center gap-3 px-4 py-3.5 sm:px-6">
        {/*
          The lockup is the way home: it clears the transcript and returns to
          the welcome screen, the way a site logo returns you to the index.
          A real <button>, not a styled div, so it is reachable by keyboard
          and announced as an action.
        */}
        <button
          type="button"
          onClick={onHome}
          title="Clear the conversation and return to the start"
          aria-label="hybrid-rag - back to the start"
          className="brut-press inline-block border-[3px] border-[var(--line)] bg-coral px-2.5 py-1 font-display text-xl font-extrabold text-ink"
        >
          hybrid-rag
        </button>
        <span className="label hidden opacity-75 sm:inline">console</span>

        <span className="ml-auto flex items-center gap-3">
          {/* Reserve the space before the mode is known, so the header does
              not jump on hydration. */}
          <span className="label px-2 py-1 min-w-[4.5rem] text-center">
            {mode === null ? (
              <span className="opacity-0">demo</span>
            ) : mode === "demo" ? (
              <span className="bg-acid px-2 py-1 text-ink">recorded</span>
            ) : (
              <span className="bg-mint px-2 py-1 text-ink">live</span>
            )}
          </span>

          <button
            type="button"
            onClick={toggleTheme}
            aria-label="Toggle colour theme"
            className="brut-sm brut-press p-2.5"
          >
            <Sun className="theme-icon-sun h-[1.15rem] w-[1.15rem]" aria-hidden />
            <Moon className="theme-icon-moon h-[1.15rem] w-[1.15rem]" aria-hidden />
          </button>
        </span>
      </div>
    </header>
  );
}
