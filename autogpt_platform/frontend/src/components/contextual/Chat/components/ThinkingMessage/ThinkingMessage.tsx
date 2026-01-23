import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";

export interface ThinkingMessageProps {
  className?: string;
}

export function ThinkingMessage({ className }: ThinkingMessageProps) {
  const [showSlowLoader, setShowSlowLoader] = useState(false);
  const [showCoffeeMessage, setShowCoffeeMessage] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const coffeeTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (timerRef.current === null) {
      timerRef.current = setTimeout(() => {
        setShowSlowLoader(true);
      }, 8000);
    }

    if (coffeeTimerRef.current === null) {
      coffeeTimerRef.current = setTimeout(() => {
        setShowCoffeeMessage(true);
      }, 10000);
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
                <span className="inline-block animate-shimmer bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-[length:200%_100%] bg-clip-text text-transparent">
                  This could take a few minutes, grab a coffee ☕️
                </span>
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
