import React, { useState } from "react";
import { Text } from "@/components/atoms/Text/Text";
import { Gear, CaretDown, CaretRight } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import type { ToolArguments } from "@/types/chat";

export interface ToolCallMessageProps {
  toolId: string;
  toolName: string;
  arguments?: ToolArguments;
  className?: string;
}

export function ToolCallMessage({
  toolId,
  toolName,
  arguments: args,
  className,
}: ToolCallMessageProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className={cn(
        "flex gap-3 rounded-lg border border-purple-200 bg-purple-50 p-4 dark:border-purple-900 dark:bg-purple-950",
        className,
      )}
    >
      {/* Icon */}
      <div className="flex-shrink-0">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-purple-500">
          <Gear size={20} weight="bold" className="animate-spin text-white" />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 space-y-2">
        <div className="flex items-center gap-2">
          <Text
            variant="body"
            className="font-semibold text-purple-900 dark:text-purple-100"
          >
            Executing: {toolName}
          </Text>
        </div>

        {/* Expandable arguments */}
        {args && Object.keys(args).length > 0 && (
          <div>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center gap-1 text-sm text-purple-700 hover:text-purple-900 dark:text-purple-300 dark:hover:text-purple-100"
            >
              {isExpanded ? (
                <CaretDown size={16} weight="bold" />
              ) : (
                <CaretRight size={16} weight="bold" />
              )}
              {isExpanded ? "Hide" : "Show"} parameters
            </button>

            {isExpanded && (
              <pre className="mt-2 overflow-x-auto rounded-md bg-purple-100 p-3 text-xs text-purple-900 dark:bg-purple-900 dark:text-purple-100">
                {JSON.stringify(args, null, 2)}
              </pre>
            )}
          </div>
        )}

        <Text variant="small" className="text-purple-600 dark:text-purple-400">
          Tool ID: {toolId.slice(0, 8)}...
        </Text>
      </div>
    </div>
  );
}
