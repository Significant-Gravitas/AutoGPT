"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { FINISHING_LINES } from "../helpers";

interface Props {
  headline: string | null;
  stage: number;
}

// The wait after a call: what ended it, then the stage its words are at,
// each stage read out once as it is reached.
export function CallFinishing({ headline, stage }: Props) {
  return (
    <div className="flex w-full max-w-md flex-col items-center gap-3">
      {headline && (
        <Text variant="body" align="center">
          {headline}
        </Text>
      )}
      <div role="status" className="flex flex-col items-center gap-2">
        <Text variant="body" tone="muted" align="center">
          {FINISHING_LINES[stage]}
        </Text>
        <ol aria-hidden className="flex items-center gap-1.5">
          {FINISHING_LINES.map((line, index) => (
            <li
              key={line}
              className={cn(
                "h-1.5 rounded-full transition-all duration-500 motion-reduce:transition-none",
                index < stage && "w-1.5 bg-foreground-subtle",
                index === stage && "w-5 bg-foreground-body",
                index > stage && "w-1.5 bg-surface-active",
              )}
            />
          ))}
        </ol>
      </div>
    </div>
  );
}
