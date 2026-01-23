import type { ToolResult } from "@/types/chat";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";
import { formatToolResponse } from "./helpers";

export interface ToolResponseMessageProps {
  toolId?: string;
  toolName: string;
  result?: ToolResult;
  success?: boolean;
  className?: string;
}

export function ToolResponseMessage({
  toolId: _toolId,
  toolName,
  result,
  success: _success,
  className,
}: ToolResponseMessageProps) {
  const formattedText = formatToolResponse(result, toolName);

  return (
    <AIChatBubble className={className}>
      <MarkdownContent content={formattedText} />
    </AIChatBubble>
  );
}
