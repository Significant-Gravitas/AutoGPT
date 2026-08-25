"use client";

import { useEffect, useState } from "react";
import { useCompactionProgress } from "./useCompactionProgress";

interface Props {
  done: boolean;
}

/** Progress bar for a running context-compaction row. Rows that mount
 *  already done (history replay) never show it; a live row fills to 100%
 *  on completion and fades out shortly after. */
export function CompactionProgress({ done }: Props) {
  const progress = useCompactionProgress(done);
  const [hidden, setHidden] = useState(done);

  useEffect(() => {
    if (!done || progress < 1 || hidden) return;
    const timer = setTimeout(() => setHidden(true), 900);
    return () => clearTimeout(timer);
  }, [done, progress, hidden]);

  if (hidden) return null;

  return (
    <div
      role="progressbar"
      aria-label="Summarizing earlier messages"
      aria-valuenow={Math.round(progress * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
      className={
        "mt-1.5 h-1 w-44 overflow-hidden rounded-full bg-zinc-100 transition-opacity duration-500 " +
        (progress >= 1 ? "opacity-0" : "opacity-100")
      }
    >
      <div
        className="h-full rounded-full bg-amber-400"
        style={{ width: `${progress * 100}%` }}
      />
    </div>
  );
}
