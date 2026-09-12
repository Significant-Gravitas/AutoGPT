"use client";

import { Fragment } from "react";
import { useTextReveal } from "../../useTextReveal";

interface Segment {
  text: string;
  bold: boolean;
}

/** Minimal inline markup: "**chunk**" renders as bold while it reveals. */
function parseBoldSegments(text: string): Segment[] {
  return text
    .split("**")
    .map((chunk, index) => ({ text: chunk, bold: index % 2 === 1 }))
    .filter((segment) => segment.text.length > 0);
}

/** Reveals the scripted text character by character, like a live LLM stream. */
export function TourStreamingText({ text }: { text: string }) {
  const segments = parseBoldSegments(text);
  const plainText = segments.map((segment) => segment.text).join("");
  const { visibleText, isDone } = useTextReveal(plainText);

  let remaining = Array.from(visibleText).length;
  const revealed = segments.map((segment, index) => {
    const chars = Array.from(segment.text);
    const visible = chars.slice(0, Math.max(0, remaining)).join("");
    remaining -= chars.length;
    if (!visible) return null;
    return segment.bold ? (
      <strong key={index}>{visible}</strong>
    ) : (
      <Fragment key={index}>{visible}</Fragment>
    );
  });

  return (
    <p>
      {revealed}
      {!isDone && (
        <span className="ml-0.5 inline-block h-4 w-2 animate-pulse rounded-sm bg-zinc-300 align-middle" />
      )}
    </p>
  );
}
