import * as React from "react";
import { cn } from "@/lib/utils";

interface Props {
  max: number;
  value: string;
  children: React.ReactNode;
}

export function CharCountedTextarea({ max, value, children }: Props) {
  const length = value.length;
  const over = length > max;
  return (
    <div className="relative">
      <span
        aria-hidden
        data-testid="char-count"
        className={cn(
          "pointer-events-none absolute right-0 top-0 z-10 text-xs tabular-nums",
          over ? "text-rose-600" : "text-zinc-400",
        )}
      >
        {length} / {max}
      </span>
      {children}
    </div>
  );
}
