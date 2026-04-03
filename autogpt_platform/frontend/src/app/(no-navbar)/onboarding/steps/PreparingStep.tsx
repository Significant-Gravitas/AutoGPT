"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Text } from "@/components/atoms/Text/Text";
import { TypingText } from "@/components/molecules/TypingText/TypingText";
import { cn } from "@/lib/utils";
import { Check } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";

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
  const [started, setStarted] = useState(false);
  const [completedItems, setCompletedItems] = useState(0);
  const [progress, setProgress] = useState(0);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    const timer = setTimeout(() => setStarted(true), 300);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!started) return;

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
        onCompleteRef.current();
      }
    }, 50);

    return () => clearInterval(progressInterval);
  }, [started]);

  return (
    <div className="flex w-full max-w-md flex-col items-center gap-8 px-4">
      <div className="flex flex-col items-center gap-4">
        <AutoGPTLogo
          className="relative right-[3rem] h-24 w-[12rem]"
          hideText
        />
        <Text variant="h3" className="text-center">
          <TypingText
            text="Preparing your workspace..."
            active={started}
            delay={400}
            speed={60}
          />
        </Text>
      </div>

      <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
        <div
          className="h-full rounded-full bg-purple-500 transition-all duration-100 ease-linear"
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
                  ? "bg-neutral-900 text-white"
                  : "bg-gray-200 text-gray-400",
              )}
            >
              <Check size={14} weight="bold" />
            </div>
            <Text
              variant="body"
              as="span"
              className={cn(
                "transition-colors",
                i < completedItems ? "!text-black" : "!text-zinc-500",
              )}
            >
              {item}
            </Text>
          </li>
        ))}
      </ul>
    </div>
  );
}
