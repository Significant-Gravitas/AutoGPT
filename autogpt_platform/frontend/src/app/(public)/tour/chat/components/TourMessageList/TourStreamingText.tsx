"use client";

import { useTextReveal } from "../../useTextReveal";

/** Reveals the scripted text character by character, like a live LLM stream.
 * The revealed text stays a single text node so test matchers and copy/paste
 * see the full sentence once the reveal completes. */
export function TourStreamingText({ text }: { text: string }) {
  const { visibleText, isDone } = useTextReveal(text);

  return (
    <p>
      {visibleText}
      {!isDone && (
        <span className="ml-0.5 inline-block h-4 w-2 animate-pulse rounded-sm bg-zinc-300 align-middle" />
      )}
    </p>
  );
}
