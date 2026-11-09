"use client";

import { Text } from "@/components/atoms/Text/Text";
import { useEffect, useRef } from "react";
import type { CaptionLine } from "../helpers";

interface Props {
  lines: CaptionLine[];
}

// The call as it is heard, Otto's words and the person's, newest at the
// bottom.
export function CallCaptions({ lines }: Props) {
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const list = listRef.current;
    if (list) list.scrollTop = list.scrollHeight;
  }, [lines]);

  if (lines.length === 0) return null;

  return (
    <div
      ref={listRef}
      role="log"
      aria-live="polite"
      aria-label="Call captions"
      className="flex max-h-64 w-full flex-col gap-2 overflow-y-auto rounded-xl border border-border-base bg-surface px-4 py-3"
    >
      {lines.map((line) => (
        <div key={line.id} className="flex gap-2">
          <Text
            variant="body-medium"
            as="span"
            tone={line.who === "otto" ? "brand" : "secondary"}
            className="shrink-0"
          >
            {line.who === "otto" ? "Otto" : "You"}
          </Text>
          <Text variant="body" as="span">
            {line.text}
          </Text>
        </div>
      ))}
    </div>
  );
}
