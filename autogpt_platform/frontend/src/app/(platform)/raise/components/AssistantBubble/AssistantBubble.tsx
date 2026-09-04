"use client";

import { useEffect, useState } from "react";

type Props = {
  text: string;
  typingDelayMs?: number;
};

export function AssistantBubble({ text, typingDelayMs = 350 }: Props) {
  const [revealed, setRevealed] = useState(typingDelayMs === 0);

  useEffect(() => {
    if (typingDelayMs === 0) return;
    const timer = setTimeout(() => setRevealed(true), typingDelayMs);
    return () => clearTimeout(timer);
  }, [typingDelayMs]);

  return (
    <div className="max-w-[80%] self-start rounded-3xl rounded-bl-lg bg-background px-5 py-3.5 text-[15px] leading-relaxed text-foreground shadow-sm duration-500 animate-in fade-in slide-in-from-bottom-3 fill-mode-both motion-reduce:animate-none">
      {revealed ? (
        text
      ) : (
        <span className="flex gap-1" aria-label="Typing">
          <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/40 [animation-delay:-0.3s]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/40 [animation-delay:-0.15s]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/40" />
        </span>
      )}
    </div>
  );
}
