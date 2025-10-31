import { cn } from "@/lib/utils";
import { Robot } from "@phosphor-icons/react";
import { MessageBubble } from "@/components/atoms/MessageBubble/MessageBubble";
import { useStreamingMessage } from "./useStreamingMessage";

export interface StreamingMessageProps {
  chunks: string[];
  className?: string;
  onComplete?: () => void;
}

export function StreamingMessage({
  chunks,
  className,
  onComplete,
}: StreamingMessageProps) {
  const { displayText } = useStreamingMessage({ chunks, onComplete });

  return (
    <div className={cn("flex gap-3 px-4 py-4", className)}>
      {/* Avatar */}
      <div className="flex-shrink-0">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-violet-600 dark:bg-violet-500">
          <Robot className="h-5 w-5 text-white" />
        </div>
      </div>

      {/* Message Content */}
      <div className="flex max-w-[70%] flex-col">
        <MessageBubble variant="assistant">
          <div className="whitespace-pre-wrap">{displayText}</div>
        </MessageBubble>

        {/* Timestamp */}
        <span className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
          Typing...
        </span>
      </div>
    </div>
  );
}
