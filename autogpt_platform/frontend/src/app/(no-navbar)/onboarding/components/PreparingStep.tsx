"use client";

import { Check } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { useEffect, useState } from "react";

const CHECKLIST = [
  "Personalizing your experience",
  "Connecting automation engines",
  "Building your space",
] as const;

const STEP_DURATION_MS = 4000;
const STEP_INTERVAL = STEP_DURATION_MS / CHECKLIST.length;

interface Props {
  onComplete: () => void;
}

export function PreparingStep({ onComplete }: Props) {
  const [completedItems, setCompletedItems] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const startTime = Date.now();

    const progressInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const pct = Math.min(100, (elapsed / STEP_DURATION_MS) * 100);
      setProgress(pct);

      const items = Math.min(
        CHECKLIST.length,
        Math.floor(elapsed / STEP_INTERVAL) + 1,
      );
      setCompletedItems(items);

      if (elapsed >= STEP_DURATION_MS) {
        clearInterval(progressInterval);
        onComplete();
      }
    }, 50);

    return () => clearInterval(progressInterval);
  }, [onComplete]);

  return (
    <div className="flex w-full max-w-md flex-col items-center gap-8 px-4">
      <div className="flex flex-col items-center gap-4">
        <Image
          src="/autogpt-logo-light-bg.png"
          alt="AutoGPT"
          width={64}
          height={64}
        />
        <h1 className="text-2xl font-semibold tracking-tight">
          Preparing your workspace...
        </h1>
      </div>

      <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
        <div
          className="h-full rounded-full bg-primary transition-all duration-100 ease-linear"
          style={{ width: `${progress}%` }}
        />
      </div>

      <ul className="flex flex-col gap-3">
        {CHECKLIST.map((item, i) => (
          <li key={item} className="flex items-center gap-3">
            <div
              className={cn(
                "flex h-6 w-6 items-center justify-center rounded-full transition-colors",
                i < completedItems
                  ? "bg-primary text-primary-foreground"
                  : "bg-gray-200 text-gray-400",
              )}
            >
              <Check size={14} weight="bold" />
            </div>
            <span
              className={cn(
                "text-sm transition-colors",
                i < completedItems
                  ? "text-foreground"
                  : "text-muted-foreground",
              )}
            >
              {item}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
