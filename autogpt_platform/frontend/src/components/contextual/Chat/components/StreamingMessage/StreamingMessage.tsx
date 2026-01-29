import { cn } from "@/lib/utils";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";
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
        <div className="flex min-w-0 flex-1 flex-col">
          <AIChatBubble>
            <MarkdownContent content={displayText} />
          </AIChatBubble>
        </div>
      </div>
    </div>
  );
}
