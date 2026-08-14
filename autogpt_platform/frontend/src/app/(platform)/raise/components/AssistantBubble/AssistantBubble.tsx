"use client";

import { useEffect, useState } from "react";

interface Props {
  text: string;
  typingDelayMs?: number;
}

export function AssistantBubble({ text, typingDelayMs = 350 }: Props) {
  const [revealed, setRevealed] = useState(typingDelayMs === 0);

  useEffect(() => {
    if (typingDelayMs === 0) return;
    const timer = setTimeout(() => setRevealed(true), typingDelayMs);
    return () => clearTimeout(timer);
  }, [typingDelayMs]);

  return (
    <div className="max-w-[80%] self-start rounded-3xl rounded-bl-lg bg-white px-5 py-3.5 text-[15px] leading-relaxed text-zinc-800 shadow-sm">
      {revealed ? (
        text
      ) : (
        <span className="flex gap-1" aria-label="Typing">
          <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-300 [animation-delay:-0.3s]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-300 [animation-delay:-0.15s]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-300" />
        </span>
      )}
    </div>
  );
}
