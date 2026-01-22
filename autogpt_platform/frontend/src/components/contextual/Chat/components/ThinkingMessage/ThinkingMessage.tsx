import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";
import { ChatLoader } from "../ChatLoader/ChatLoader";

export interface ThinkingMessageProps {
  className?: string;
}

export function ThinkingMessage({ className }: ThinkingMessageProps) {
  const [showSlowLoader, setShowSlowLoader] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (timerRef.current === null) {
      timerRef.current = setTimeout(() => {
        setShowSlowLoader(true);
      }, 8000);
    }

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
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
              {showSlowLoader ? (
                <ChatLoader />
              ) : (
                <span
                  className="inline-block bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-clip-text text-transparent"
                  style={{
                    backgroundSize: "200% 100%",
                    animation: "shimmer 2s ease-in-out infinite",
                  }}
                >
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
