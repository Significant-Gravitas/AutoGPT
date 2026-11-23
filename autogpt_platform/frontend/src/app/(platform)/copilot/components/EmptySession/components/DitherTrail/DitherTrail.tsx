"use client";

import { useEffect, useRef } from "react";
import { runDitherTrail } from "./ditherField";

/** A pale trail of dither squares that follows the mouse, behind the
 *  page's content. It is decoration only: nothing is drawn on a touch
 *  screen, which has no pointer to follow, or when the reader has asked for
 *  less motion. */
export function DitherTrail() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const hasMouse = window.matchMedia(
      "(hover: hover) and (pointer: fine)",
    ).matches;
    const wantsStill = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (!hasMouse || wantsStill) return;
    return runDitherTrail(canvas, context);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none absolute inset-0 z-0 size-full text-foreground-ink"
    />
  );
}
