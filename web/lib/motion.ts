"use client";

import { useEffect, useLayoutEffect } from "react";

/**
 * `useLayoutEffect` in the browser, `useEffect` during prerender.
 *
 * GSAP entrance animations have to set their start state before the browser
 * paints, or the element flashes at its final position for one frame and then
 * jumps back to animate. `useLayoutEffect` is the hook that runs early enough -
 * but React warns when it runs during server rendering, and this app is a
 * static export, so every component is prerendered at build time.
 */
export const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

/**
 * The house easing and durations.
 *
 * Matched to `.brut-press` in `globals.css`, which uses
 * `cubic-bezier(0.2, 0, 0, 1)` over 120ms. Entrances are slower than a press
 * because they cover more distance, but the curve is the same so a click and a
 * reveal feel like the same hand.
 */
export const EASE = "power3.out";
export const ENTER = 0.55;
export const STAGGER = 0.06;
