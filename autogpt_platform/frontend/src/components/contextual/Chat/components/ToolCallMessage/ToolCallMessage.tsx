import { Text } from "@/components/atoms/Text/Text";
import type { ToolArguments } from "@/types/chat";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";

export interface ToolCallMessageProps {
  toolId?: string;
  toolName: string;
  arguments?: ToolArguments;
  className?: string;
}

export function ToolCallMessage({
  toolId,
  toolName,
  arguments: toolArguments,
  className,
}: ToolCallMessageProps) {
  const displayKey = toolName || toolId;

  const displayData = toolArguments
    ? JSON.stringify(toolArguments)
    : "No arguments";

  const displayText = `${displayKey}: ${displayData}`;

  return (
    <AIChatBubble className={className}>
      <Text variant="small" className="text-neutral-500">
        {displayText}
      </Text>
    </AIChatBubble>
  );
}
