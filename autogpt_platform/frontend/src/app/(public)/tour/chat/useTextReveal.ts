"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { useRef, useState } from "react";
import { REVEAL_CHARS_PER_TICK, REVEAL_TICK_MS } from "./helpers";

/** Types `text` out character by character, like a live LLM stream. Callers
 * re-key their component to restart the reveal for a new text. */
export function useTextReveal(text: string) {
  // Code points, not UTF-16 units — slicing mid-surrogate would briefly
  // render a broken glyph for emoji like 🎉.
  const chars = useRef(Array.from(text));
  const [visibleCount, setVisibleCount] = useState(0);

  useMountEffect(() => {
    const id = setInterval(() => {
      setVisibleCount((count) => {
        const next = Math.min(
          count + REVEAL_CHARS_PER_TICK,
          chars.current.length,
        );
        if (next >= chars.current.length) clearInterval(id);
        return next;
      });
    }, REVEAL_TICK_MS);
    return () => clearInterval(id);
  });

  return {
    visibleText: chars.current.slice(0, visibleCount).join(""),
    isDone: visibleCount >= chars.current.length,
  };
}
