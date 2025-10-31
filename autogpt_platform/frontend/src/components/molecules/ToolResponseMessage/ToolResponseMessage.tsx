import React, { useState } from "react";
import { Text } from "@/components/atoms/Text/Text";
import {
  CheckCircle,
  XCircle,
  CaretDown,
  CaretRight,
} from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import type { ToolResult } from "@/types/chat";

export interface ToolResponseMessageProps {
  toolId: string;
  toolName: string;
  result: ToolResult;
  success?: boolean;
  className?: string;
}

export function ToolResponseMessage({
  toolId,
  toolName,
  result,
  success = true,
  className,
}: ToolResponseMessageProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const isObjectResult = typeof result === "object";
  const resultString = isObjectResult
    ? JSON.stringify(result, null, 2)
    : result;
  const shouldTruncate = resultString.length > 200;

  return (
    <div
      className={cn(
        "flex gap-3 rounded-lg border p-4",
        success
          ? "border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-950"
          : "border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950",
        className,
      )}
    >
      {/* Icon */}
      <div className="flex-shrink-0">
        <div
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-full",
            success ? "bg-green-500" : "bg-red-500",
          )}
        >
          {success ? (
            <CheckCircle size={20} weight="fill" className="text-white" />
          ) : (
            <XCircle size={20} weight="fill" className="text-white" />
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 space-y-2">
        <div className="flex items-center gap-2">
          <Text
            variant="body"
            className={cn(
              "font-semibold",
              success
                ? "text-green-900 dark:text-green-100"
                : "text-red-900 dark:text-red-100",
            )}
          >
            {success ? "Completed" : "Failed"}: {toolName}
          </Text>
        </div>

        {/* Result */}
        <div>
          {shouldTruncate && !isExpanded ? (
            <>
              <pre
                className={cn(
                  "overflow-x-auto rounded-md p-3 text-xs",
                  success
                    ? "bg-green-100 text-green-900 dark:bg-green-900 dark:text-green-100"
                    : "bg-red-100 text-red-900 dark:bg-red-900 dark:text-red-100",
                )}
              >
                {resultString.slice(0, 200)}...
              </pre>
              <button
                onClick={() => setIsExpanded(true)}
                className={cn(
                  "mt-2 flex items-center gap-1 text-sm",
                  success
                    ? "text-green-700 hover:text-green-900 dark:text-green-300 dark:hover:text-green-100"
                    : "text-red-700 hover:text-red-900 dark:text-red-300 dark:hover:text-red-100",
                )}
              >
                <CaretDown size={16} weight="bold" />
                Show full result
              </button>
            </>
          ) : (
            <>
              <pre
                className={cn(
                  "overflow-x-auto rounded-md p-3 text-xs",
                  success
                    ? "bg-green-100 text-green-900 dark:bg-green-900 dark:text-green-100"
                    : "bg-red-100 text-red-900 dark:bg-red-900 dark:text-red-100",
                )}
              >
                {resultString}
              </pre>
              {shouldTruncate && isExpanded && (
                <button
                  onClick={() => setIsExpanded(false)}
                  className={cn(
                    "mt-2 flex items-center gap-1 text-sm",
                    success
                      ? "text-green-700 hover:text-green-900 dark:text-green-300 dark:hover:text-green-100"
                      : "text-red-700 hover:text-red-900 dark:text-red-300 dark:hover:text-red-100",
                  )}
                >
                  <CaretRight size={16} weight="bold" />
                  Show less
                </button>
              )}
            </>
          )}
        </div>

        <Text
          variant="small"
          className={
            success
              ? "text-green-600 dark:text-green-400"
              : "text-red-600 dark:text-red-400"
          }
        >
          Tool ID: {toolId.slice(0, 8)}...
        </Text>
      </div>
    </div>
  );
}
