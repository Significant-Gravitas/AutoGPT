import { cn } from "@/lib/utils";
import { RobotIcon } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import { MessageBubble } from "../MessageBubble/MessageBubble";

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
        <div className="flex-shrink-0">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-500">
            <RobotIcon className="h-4 w-4 text-indigo-50" />
          </div>
        </div>

        <div className="flex min-w-0 flex-1 flex-col">
          <MessageBubble variant="assistant">
            <div className="transition-all duration-500 ease-in-out">
              {showSlowLoader ? (
                <div className="flex flex-col items-center gap-3 py-2">
                  <div className="loader" style={{ flexShrink: 0 }} />
                  <p className="text-sm text-slate-700">
                    Taking a bit longer to think, wait a moment please
                  </p>
                </div>
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
          </MessageBubble>
        </div>
      </div>
    </div>
  );
}
