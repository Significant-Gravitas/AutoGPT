import React, { useState } from "react";
import { Text } from "@/components/atoms/Text/Text";
import { Wrench, Spinner, CaretDown, CaretUp } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { getToolDisplayName } from "@/app/(platform)/chat/helpers";
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
        "overflow-hidden rounded-lg border transition-all duration-200",
        "border-neutral-200 dark:border-neutral-700",
        "bg-white dark:bg-neutral-900",
        "animate-in fade-in-50 slide-in-from-top-1",
        className,
      )}
    >
      {/* Header */}
      <div
        className={cn(
          "flex items-center justify-between px-3 py-2",
          "bg-gradient-to-r from-neutral-50 to-neutral-100 dark:from-neutral-800/20 dark:to-neutral-700/20",
        )}
      >
        <div className="flex items-center gap-2">
          <Wrench
            size={16}
            weight="bold"
            className="text-neutral-500 dark:text-neutral-400"
          />
          <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
            {getToolDisplayName(toolName)}
          </span>
          <div className="ml-2 flex items-center gap-1.5">
            <Spinner
              size={16}
              weight="bold"
              className="animate-spin text-blue-500"
            />
            <span className="text-xs text-neutral-500 dark:text-neutral-400">
              Executing...
            </span>
          </div>
        </div>

        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="rounded p-1 hover:bg-neutral-200/50 dark:hover:bg-neutral-700/50"
          aria-label={isExpanded ? "Collapse details" : "Expand details"}
        >
          {isExpanded ? (
            <CaretUp
              size={16}
              weight="bold"
              className="text-neutral-600 dark:text-neutral-400"
            />
          ) : (
            <CaretDown
              size={16}
              weight="bold"
              className="text-neutral-600 dark:text-neutral-400"
            />
          )}
        </button>
      </div>

      {/* Expandable Content */}
      {isExpanded && (
        <div className="px-4 py-3">
          {args && Object.keys(args).length > 0 && (
            <div className="mb-3">
              <div className="mb-2 text-xs font-medium text-neutral-600 dark:text-neutral-400">
                Parameters:
              </div>
              <div className="rounded-md bg-neutral-50 p-3 dark:bg-neutral-800">
                <pre className="overflow-x-auto text-xs text-neutral-700 dark:text-neutral-300">
                  {JSON.stringify(args, null, 2)}
                </pre>
              </div>
            </div>
          )}

          <Text
            variant="small"
            className="text-neutral-500 dark:text-neutral-400"
          >
            Tool ID: {toolId.slice(0, 8)}...
          </Text>
        </div>
      )}
    </div>
  );
}
