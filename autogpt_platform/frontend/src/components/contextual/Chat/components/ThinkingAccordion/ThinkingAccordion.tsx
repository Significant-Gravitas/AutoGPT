"use client";

import { TypeWriter } from "@/components/molecules/Typewriter/Typewriter";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { ThinkingAccordionAnimation } from "./components/ThinkingAccordionAnimation";

const THINKING_LABELS = [
  "Hmm, let me think…",
  "Cooking up something good…",
  "Connecting the dots…",
  "Brewing ideas…",
  "Down the rabbit hole…",
  "Following the thread…",
  "Almost there, maybe…",
  "One sec, got an idea…",
  "Piecing it together…",
  "Digging deeper…",
  "Let me check that…",
  "Working on it…",
  "Hold that thought…",
  "Getting there…",
] as const;

const LABEL_ROTATION_INTERVAL = 4000;

export interface Props {
  className?: string;
}

export function ThinkingAccordion({ className }: Props) {
  const [currentLabel, setCurrentLabel] = useState<string>(
    () => THINKING_LABELS[Math.floor(Math.random() * THINKING_LABELS.length)],
  );

  // Rotate labels randomly
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentLabel((prev) => {
        let next: string;
        do {
          next =
            THINKING_LABELS[Math.floor(Math.random() * THINKING_LABELS.length)];
        } while (next === prev);
        return next;
      });
    }, LABEL_ROTATION_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      className={cn(
        "group relative flex w-full justify-start gap-3 px-4 py-3",
        className,
      )}
    >
      <div className="flex w-full max-w-3xl gap-3">
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="inline-flex w-fit items-center gap-2 rounded-md px-1 py-2">
            <ThinkingAccordionAnimation />
            <TypeWriter
              text={currentLabel}
              className="inline-block animate-shimmer bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-[length:200%_100%] bg-clip-text text-sm font-medium text-transparent"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
