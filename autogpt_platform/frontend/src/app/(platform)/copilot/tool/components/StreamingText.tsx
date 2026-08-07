"use client";

import { useEffect, useRef, useState } from "react";

export function StreamingText({ text }: { text: string }) {
  const [shown, setShown] = useState("");
  // Continues from the last revealed index so growing deltas don't restart the typer.
  const revealedRef = useRef(0);

  useEffect(() => {
    if (revealedRef.current >= text.length) return;
    const id = setInterval(() => {
      revealedRef.current = Math.min(revealedRef.current + 2, text.length);
      setShown(text.slice(0, revealedRef.current));
      if (revealedRef.current >= text.length) clearInterval(id);
    }, 9);
    return () => clearInterval(id);
  }, [text]);

  const streaming = shown.length < text.length;

  return (
    <p className="whitespace-pre-wrap text-[1rem] leading-relaxed text-slate-900">
      {shown}
      <span
        aria-hidden
        className={
          "ml-0.5 inline-block h-[1.05em] w-2 bg-zinc-900 align-text-bottom motion-reduce:animate-none " +
          (streaming ? "animate-caret-blink" : "")
        }
      />
    </p>
  );
}
