import { cn } from "@/lib/utils";
import { Robot } from "@phosphor-icons/react";
import { MessageBubble } from "../MessageBubble/MessageBubble";

export interface ThinkingMessageProps {
  className?: string;
}

export function ThinkingMessage({ className }: ThinkingMessageProps) {
  return (
    <div
      className={cn(
        "group relative flex w-full justify-start gap-3 px-4 py-3",
        className,
      )}
    >
      <div className="flex w-full max-w-3xl gap-3">
        <div className="flex-shrink-0">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-neutral-200">
            <Robot className="h-4 w-4 text-neutral-600" />
          </div>
        </div>

        <div className="flex min-w-0 flex-1 flex-col">
          <MessageBubble variant="assistant">
            <span
              className="inline-block bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-clip-text text-transparent"
              style={{
                backgroundSize: "200% 100%",
                animation: "shimmer 2s ease-in-out infinite",
              }}
            >
              Thinking...
            </span>
          </MessageBubble>
        </div>
      </div>
    </div>
  );
}
