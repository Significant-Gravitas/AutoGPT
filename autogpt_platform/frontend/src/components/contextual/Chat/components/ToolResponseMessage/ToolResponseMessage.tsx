import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { ToolResult } from "@/types/chat";
import { WarningCircleIcon } from "@phosphor-icons/react";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";
import {
  formatToolResponse,
  getErrorMessage,
  isErrorResponse,
} from "./helpers";

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
  if (isErrorResponse(result)) {
    const errorMessage = getErrorMessage(result);
    return (
      <AIChatBubble className={className}>
        <div className="flex items-center gap-2">
          <WarningCircleIcon
            size={14}
            weight="regular"
            className="shrink-0 text-neutral-400"
          />
          <Text variant="small" className={cn("text-xs text-neutral-500")}>
            {errorMessage}
          </Text>
        </div>
      </AIChatBubble>
    );
  }

  const formattedText = formatToolResponse(result, toolName);

  return (
    <AIChatBubble className={className}>
      <MarkdownContent content={formattedText} />
    </AIChatBubble>
  );
}
