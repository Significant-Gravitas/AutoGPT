import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { ToolArguments } from "@/types/chat";
import type { Block } from "@/lib/autogpt-server-api/types";
import { useGetV1ListAvailableBlocks } from "@/app/api/__generated__/endpoints/blocks/blocks";
import { useMemo } from "react";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";
import {
  formatToolArguments,
  getToolActionPhrase,
  getToolIcon,
} from "./helpers";

export interface ToolCallMessageProps {
  toolId?: string;
  toolName: string;
  arguments?: ToolArguments;
  isStreaming?: boolean;
  className?: string;
}

export function ToolCallMessage({
  toolName,
  arguments: toolArguments,
  isStreaming = false,
  className,
}: ToolCallMessageProps) {
  // Fetch blocks only when needed for run_block tool
  const { data: blocksResponse } = useGetV1ListAvailableBlocks({
    query: {
      enabled: toolName === "run_block",
      staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    },
  });

  // Create a memoized map of block IDs to names
  const blocksById = useMemo(() => {
    if (!blocksResponse?.data) return undefined;
    const blocks = blocksResponse.data as Block[];
    return new Map(blocks.map((block) => [block.id, block.name]));
  }, [blocksResponse?.data]);

  const actionPhrase = getToolActionPhrase(toolName);
  const argumentsText = formatToolArguments(
    toolName,
    toolArguments,
    blocksById,
  );
  const displayText = `${actionPhrase}${argumentsText}`;
  const IconComponent = getToolIcon(toolName);

  return (
    <AIChatBubble className={className}>
      <div className="flex items-center gap-2">
        <IconComponent
          size={14}
          weight={isStreaming ? "regular" : "regular"}
          className={cn(
            "shrink-0",
            isStreaming ? "text-neutral-500" : "text-neutral-400",
          )}
        />
        <Text
          variant="small"
          className={cn(
            "text-xs",
            isStreaming
              ? "bg-gradient-to-r from-neutral-600 via-neutral-500 to-neutral-600 bg-[length:200%_100%] bg-clip-text text-transparent [animation:shimmer_2s_ease-in-out_infinite]"
              : "text-neutral-500",
          )}
        >
          {displayText}
        </Text>
      </div>
    </AIChatBubble>
  );
}
