import { cn } from "@/lib/utils";
import { Robot } from "@phosphor-icons/react";
import { MessageBubble } from "@/app/(platform)/chat/components/MessageBubble/MessageBubble";
import { MarkdownContent } from "@/app/(platform)/chat/components/MarkdownContent/MarkdownContent";
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
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-purple-600 dark:bg-purple-500">
          <Robot className="h-5 w-5 text-white" />
        </div>
      </div>

      {/* Message Content */}
      <div className="flex max-w-[70%] flex-col">
        <MessageBubble variant="assistant">
          <MarkdownContent content={displayText} />
        </MessageBubble>

        {/* Timestamp */}
        <span className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
          Typing...
        </span>
      </div>
    </div>
  );
}
