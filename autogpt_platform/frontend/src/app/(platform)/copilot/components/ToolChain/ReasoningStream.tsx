"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface Props {
  text: string;
  live: boolean;
}

export function ReasoningStream({ text, live }: Props) {
  const viewportRef = useRef<HTMLDivElement>(null);

  // Auto-scroll pinned to the newest line while thinking streams.
  useEffect(() => {
    const el = viewportRef.current;
    if (live && el) el.scrollTop = el.scrollHeight;
  }, [text, live]);

  return (
    <div
      ref={viewportRef}
      className={cn(
        "text-body-sm leading-5 text-foreground-subtle",
        live
          ? "max-h-24 overflow-hidden [mask-image:linear-gradient(to_bottom,transparent_0,black_16px,black_100%)]"
          : "scrollbar-none max-h-64 overflow-y-auto",
      )}
    >
      <p className="whitespace-pre-wrap">{text}</p>
    </div>
  );
}
