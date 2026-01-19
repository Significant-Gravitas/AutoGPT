import { Text } from "@/components/atoms/Text/Text";
import type { ToolResult } from "@/types/chat";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";

export interface ToolResponseMessageProps {
  toolId?: string;
  toolName: string;
  result?: ToolResult;
  success?: boolean;
  className?: string;
}

export function ToolResponseMessage({
  toolId,
  toolName,
  result: _result,
  success: _success = true,
  className,
}: ToolResponseMessageProps) {
  const displayKey = toolId || toolName;
  const resultValue =
    typeof _result === "string"
      ? _result
      : _result
        ? JSON.stringify(_result)
        : toolName;
  const displayText = `${displayKey}: ${resultValue}`;

  return (
    <AIChatBubble className={className}>
      <Text variant="small" className="text-neutral-500">
        {displayText}
      </Text>
    </AIChatBubble>
  );
}
