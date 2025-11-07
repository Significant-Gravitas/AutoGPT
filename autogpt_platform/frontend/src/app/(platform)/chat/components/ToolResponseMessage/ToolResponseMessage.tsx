import React, { useState } from "react";
import { Text } from "@/components/atoms/Text/Text";
import {
  CheckCircle,
  XCircle,
  CaretDown,
  CaretUp,
  Wrench,
} from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { getToolDisplayName } from "@/app/(platform)/chat/helpers";
import type { ToolResult } from "@/types/chat";

export interface ToolResponseMessageProps {
  toolId: string;
  toolName: string;
  result: ToolResult;
  success?: boolean;
  className?: string;
}

// Check if result should be hidden (special response types)
function shouldHideResult(result: ToolResult): boolean {
  try {
    const resultString =
      typeof result === "string" ? result : JSON.stringify(result);
    const parsed = JSON.parse(resultString);

    // Hide raw JSON for these special types
    if (parsed.type === "agent_carousel") return true;
    if (parsed.type === "execution_started") return true;
    if (parsed.type === "setup_requirements") return true;
    if (parsed.type === "no_results") return true;

    return false;
  } catch {
    return false;
  }
}

// Get a friendly summary for special response types
function getResultSummary(result: ToolResult): string | null {
  try {
    const resultString =
      typeof result === "string" ? result : JSON.stringify(result);
    const parsed = JSON.parse(resultString);

    if (parsed.type === "agent_carousel") {
      return `Found ${parsed.agents?.length || parsed.count || 0} agents${parsed.query ? ` matching "${parsed.query}"` : ""}`;
    }
    if (parsed.type === "execution_started") {
      return `Started execution${parsed.execution_id ? ` (ID: ${parsed.execution_id.slice(0, 8)}...)` : ""}`;
    }
    if (parsed.type === "setup_requirements") {
      return "Retrieved setup requirements";
    }
    if (parsed.type === "no_results") {
      return parsed.message || "No results found";
    }

    return null;
  } catch {
    return null;
  }
}

export function ToolResponseMessage({
  toolId,
  toolName,
  result,
  success = true,
  className,
}: ToolResponseMessageProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const hideResult = shouldHideResult(result);
  const resultSummary = getResultSummary(result);
  const resultString =
    typeof result === "object"
      ? JSON.stringify(result, null, 2)
      : String(result);
  const shouldTruncate = resultString.length > 200;

  return (
    <div
      className={cn(
        "overflow-hidden rounded-lg border transition-all duration-200",
        success
          ? "border-neutral-200 dark:border-neutral-700"
          : "border-red-200 dark:border-red-800",
        "bg-white dark:bg-neutral-900",
        "animate-in fade-in-50 slide-in-from-top-1",
        className,
      )}
    >
      {/* Header */}
      <div
        className={cn(
          "flex items-center justify-between px-3 py-2",
          "bg-gradient-to-r",
          success
            ? "from-neutral-50 to-neutral-100 dark:from-neutral-800/20 dark:to-neutral-700/20"
            : "from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20",
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
            {success ? (
              <CheckCircle size={16} weight="fill" className="text-green-500" />
            ) : (
              <XCircle size={16} weight="fill" className="text-red-500" />
            )}
            <span className="text-xs text-neutral-500 dark:text-neutral-400">
              {success ? "Completed" : "Error"}
            </span>
          </div>
        </div>

        {!hideResult && (
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
        )}
      </div>

      {/* Expandable Content */}
      {isExpanded && !hideResult && (
        <div className="px-4 py-3">
          <div className="mb-2 text-xs font-medium text-neutral-600 dark:text-neutral-400">
            Result:
          </div>
          <div
            className={cn(
              "rounded-md p-3",
              success
                ? "bg-green-50 dark:bg-green-900/20"
                : "bg-red-50 dark:bg-red-900/20",
            )}
          >
            <pre
              className={cn(
                "whitespace-pre-wrap text-xs",
                success
                  ? "text-green-800 dark:text-green-200"
                  : "text-red-800 dark:text-red-200",
              )}
            >
              {shouldTruncate && !isExpanded
                ? `${resultString.slice(0, 200)}...`
                : resultString}
            </pre>
          </div>

          <Text
            variant="small"
            className="mt-2 text-neutral-500 dark:text-neutral-400"
          >
            Tool ID: {toolId.slice(0, 8)}...
          </Text>
        </div>
      )}

      {/* Summary for special response types */}
      {hideResult && resultSummary && (
        <div className="px-4 py-2">
          <Text
            variant="small"
            className="text-neutral-600 dark:text-neutral-400"
          >
            {resultSummary}
          </Text>
        </div>
      )}
    </div>
  );
}
