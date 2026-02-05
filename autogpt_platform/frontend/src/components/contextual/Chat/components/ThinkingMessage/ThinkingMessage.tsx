import { Progress } from "@/components/atoms/Progress/Progress";
import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";
import { useAsymptoticProgress } from "../ToolCallMessage/useAsymptoticProgress";

export interface ThinkingMessageProps {
  className?: string;
  isComplete?: boolean;
  onAnimationComplete?: () => void;
}

export function ThinkingMessage({
  className,
  isComplete = false,
  onAnimationComplete,
}: ThinkingMessageProps) {
  const [showSlowLoader, setShowSlowLoader] = useState(false);
  const [showCoffeeMessage, setShowCoffeeMessage] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const coffeeTimerRef = useRef<NodeJS.Timeout | null>(null);
  const delayTimerRef = useRef<NodeJS.Timeout | null>(null);
  const { progress, isAnimationDone } = useAsymptoticProgress(
    showCoffeeMessage,
    isComplete,
  );

  useEffect(() => {
    if (timerRef.current === null) {
      timerRef.current = setTimeout(() => {
        setShowSlowLoader(true);
      }, 3000);
    }

    if (coffeeTimerRef.current === null) {
      coffeeTimerRef.current = setTimeout(() => {
        setShowCoffeeMessage(true);
      }, 8000);
    }

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      if (coffeeTimerRef.current) {
        clearTimeout(coffeeTimerRef.current);
        coffeeTimerRef.current = null;
      }
    };
  }, []);

  // Handle completion animation delay before unmounting
  useEffect(() => {
    if (isAnimationDone && onAnimationComplete) {
      delayTimerRef.current = setTimeout(() => {
        onAnimationComplete();
      }, 200); // 200ms delay after animation completes
    }

    return () => {
      if (delayTimerRef.current) {
        clearTimeout(delayTimerRef.current);
        delayTimerRef.current = null;
      }
    };
  }, [isAnimationDone, onAnimationComplete]);

  return (
    <div
      className={cn(
        "group relative flex w-full justify-start gap-3 px-4 py-3",
        className,
      )}
    >
      <div className="flex w-full max-w-3xl gap-3">
        <div className="flex min-w-0 flex-1 flex-col">
          <AIChatBubble>
            <div className="transition-all duration-500 ease-in-out">
              {showCoffeeMessage ? (
                <div className="flex flex-col items-center gap-3">
                  <div className="flex w-full max-w-[280px] flex-col gap-1.5">
                    <div className="flex items-center justify-between text-xs text-neutral-500">
                      <span>Working on it...</span>
                      <span>{Math.round(progress)}%</span>
                    </div>
                    <Progress value={progress} className="h-2 w-full" />
                  </div>
                  <span className="inline-block animate-shimmer bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-[length:200%_100%] bg-clip-text text-transparent">
                    This could take a few minutes, grab a coffee ☕️
                  </span>
                </div>
              ) : showSlowLoader ? (
                <span className="inline-block animate-shimmer bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-[length:200%_100%] bg-clip-text text-transparent">
                  Taking a bit more time...
                </span>
              ) : (
                <span className="inline-block animate-shimmer bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-[length:200%_100%] bg-clip-text text-transparent">
                  Thinking...
                </span>
              )}
            </div>
          </AIChatBubble>
        </div>
      </div>
    </div>
  );
}
