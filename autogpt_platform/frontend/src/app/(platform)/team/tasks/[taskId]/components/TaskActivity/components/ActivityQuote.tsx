"use client";

import { cn } from "@/lib/utils";
import { useState } from "react";
import { ActivityMarkdown } from "./ActivityMarkdown";

/** Past this many characters the quote clamps to a few lines behind a
 *  "Read more" toggle, so one chatty note can't bury the rest of the feed. */
const CLAMP_THRESHOLD = 280;

interface Props {
  text: string;
}

/** The words an actor left on the task — a note, a question, the outcome —
 *  quoted in a card under the entry's header. */
export function ActivityQuote({ text }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const isLong = text.length > CLAMP_THRESHOLD;

  return (
    <div className="mt-2 max-w-2xl rounded-xl border border-zinc-200 bg-white px-3.5 py-2.5">
      <div className={cn(!isExpanded && isLong && "line-clamp-4")}>
        <ActivityMarkdown text={text} />
      </div>
      {isLong ? (
        <button
          type="button"
          onClick={() => setIsExpanded((value) => !value)}
          className="mt-1.5 text-xs font-medium text-zinc-500 transition-colors hover:text-zinc-800"
        >
          {isExpanded ? "Show less" : "Read more"}
        </button>
      ) : null}
    </div>
  );
}
