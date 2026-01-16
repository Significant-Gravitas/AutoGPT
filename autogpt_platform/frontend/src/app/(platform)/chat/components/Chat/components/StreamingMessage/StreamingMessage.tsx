import { cn } from "@/lib/utils";
import { RobotIcon } from "@phosphor-icons/react";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";
import { MessageBubble } from "../MessageBubble/MessageBubble";
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
    <div
      className={cn(
        "group relative flex w-full justify-start gap-3 px-4 py-3",
        className,
      )}
    >
      <div className="flex w-full max-w-3xl gap-3">
        <div className="flex-shrink-0">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600">
            <RobotIcon className="h-4 w-4 text-indigo-50" />
          </div>
        </div>

        <div className="flex min-w-0 flex-1 flex-col">
          <MessageBubble variant="assistant">
            <MarkdownContent content={displayText} />
          </MessageBubble>
        </div>
      </div>
    </div>
  );
}
