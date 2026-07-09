"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { useRef, useState } from "react";

const CHARS_PER_TICK = 2;
const TICK_MS = 16;

/** Reveals the scripted text character by character, like a live LLM stream.
 * The revealed text stays a single text node so test matchers and copy/paste
 * see the full sentence once the reveal completes. */
export function TourStreamingText({ text }: { text: string }) {
  // Code points, not UTF-16 units — slicing mid-surrogate would briefly
  // render a broken glyph for emoji like 🎉.
  const chars = useRef(Array.from(text));
  const [visibleCount, setVisibleCount] = useState(0);

  useMountEffect(() => {
    const id = setInterval(() => {
      setVisibleCount((count) => {
        const next = Math.min(count + CHARS_PER_TICK, chars.current.length);
        if (next >= chars.current.length) clearInterval(id);
        return next;
      });
    }, TICK_MS);
    return () => clearInterval(id);
  });

  const isDone = visibleCount >= chars.current.length;

  return (
    <p>
      {chars.current.slice(0, visibleCount).join("")}
      {!isDone && (
        <span className="ml-0.5 inline-block h-4 w-2 animate-pulse rounded-sm bg-zinc-300 align-middle" />
      )}
    </p>
  );
}
