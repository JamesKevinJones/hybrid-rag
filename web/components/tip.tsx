"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";

/**
 * A hover/focus explanation for a term that is not self-evident.
 *
 * Rendered through a **portal** rather than positioned inside the trigger.
 * Both places this is used sit in a clipping container - the status strip is
 * `overflow-x-auto` (which clips vertically too, because CSS makes `overflow-y`
 * `auto` the moment one axis is) and the transcript is `overflow-y-auto`. An
 * absolutely positioned tooltip in either one gets cut in half.
 *
 * Opens on hover AND on focus, and the trigger is a real focusable button, so
 * this is reachable by keyboard and by tap on a touch screen where `:hover`
 * does not exist. `aria-describedby` ties the panel to the trigger, so a screen
 * reader announces the explanation rather than an unlabelled control.
 */
export function Tip({
  content,
  children,
  className = "",
}: {
  content: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  const id = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [box, setBox] = useState<{ top: number; left: number } | null>(null);

  const open = useCallback(() => {
    const node = triggerRef.current;
    if (!node) return;
    const r = node.getBoundingClientRect();
    setBox({ top: r.bottom + 8, left: r.left });
  }, []);

  const close = useCallback(() => setBox(null), []);

  // A tooltip anchored to viewport coordinates goes stale the moment anything
  // moves, and the transcript scrolls constantly.
  useEffect(() => {
    if (!box) return;
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [box, close]);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-describedby={box ? id : undefined}
        onMouseEnter={open}
        onMouseLeave={close}
        onFocus={open}
        onBlur={close}
        onClick={() => (box ? close() : open())}
        onKeyDown={(e) => e.key === "Escape" && close()}
        className={`cursor-help text-left ${className}`}
      >
        {children}
      </button>

      {box &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            id={id}
            role="tooltip"
            style={{
              top: box.top,
              // Clamp to the viewport so a tooltip on the last status cell does
              // not hang off the right edge.
              left: Math.min(box.left, window.innerWidth - 340),
            }}
            className="brut pointer-events-none fixed z-50 max-w-[20rem] px-3.5 py-3 text-[0.8125rem] leading-relaxed"
          >
            {content}
          </div>,
          document.body,
        )}
    </>
  );
}
