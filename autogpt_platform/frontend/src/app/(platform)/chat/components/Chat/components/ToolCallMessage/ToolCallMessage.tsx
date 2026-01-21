import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { WrenchIcon } from "@phosphor-icons/react";
import { getToolActionPhrase } from "../../helpers";

export interface ToolCallMessageProps {
  toolName: string;
  className?: string;
}

export function ToolCallMessage({ toolName, className }: ToolCallMessageProps) {
  return (
    <div className={cn("flex items-center justify-center gap-2", className)}>
      <WrenchIcon
        size={14}
        weight="bold"
        className="flex-shrink-0 text-neutral-500"
      />
      <Text variant="small" className="text-neutral-500">
        {getToolActionPhrase(toolName)}...
      </Text>
    </div>
  );
}
