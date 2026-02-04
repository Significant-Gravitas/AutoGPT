"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { cn } from "@/lib/utils";
import type { ToolArguments, ToolResult } from "@/types/chat";
import { useState } from "react";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";
import {
  formatToolArguments,
  formatToolResultAsJson,
  getToolActionPhrase,
  getToolIcon,
  getToolResultErrorMessage,
  getToolResultMessage,
  isToolResultError,
} from "./helpers";

/** Tool response data passed from parent */
interface ToolResponseData {
  toolId: string;
  toolName: string;
  result: ToolResult;
  success?: boolean;
}

export interface Props {
  toolId?: string;
  toolName: string;
  arguments?: ToolArguments;
  isStreaming?: boolean;
  className?: string;
  /** The corresponding tool response, if available */
  toolResponse?: ToolResponseData;
}

export function ToolCallMessage({
  toolName,
  arguments: toolArguments,
  isStreaming = false,
  className,
  toolResponse,
}: Props) {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const actionPhrase = getToolActionPhrase(toolName);
  const argumentsText = formatToolArguments(
    toolName,
    toolArguments,
    toolResponse,
  );
  const displayText = `${actionPhrase}${argumentsText}`;
  const IconComponent = getToolIcon(toolName);

  // Check if there's an error in the tool response
  const hasError = toolResponse && isToolResultError(toolResponse.result);
  const errorMessage = hasError
    ? getToolResultErrorMessage(toolResponse.result)
    : null;

  // Only make clickable if there's a tool response to show (and not an error)
  const isClickable = !!toolResponse && !isStreaming && !hasError;

  const content = (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <IconComponent
          size={14}
          weight="regular"
          className={cn(
            "shrink-0",
            hasError
              ? "text-red-400"
              : isStreaming
                ? "text-neutral-500"
                : "text-neutral-400",
          )}
        />
        <Text
          variant="small"
          className={cn(
            "text-xs",
            hasError
              ? "text-red-500"
              : isStreaming
                ? "bg-gradient-to-r from-neutral-600 via-neutral-500 to-neutral-600 bg-[length:200%_100%] bg-clip-text text-transparent [animation:shimmer_2s_ease-in-out_infinite]"
                : "text-neutral-500",
          )}
        >
          {displayText}
        </Text>
      </div>
      {errorMessage && (
        <Text variant="body-medium" className="text-md ml-6 text-red-500">
          {errorMessage}
        </Text>
      )}
    </div>
  );

  if (!isClickable) {
    return <AIChatBubble className={className}>{content}</AIChatBubble>;
  }

  const messageContent = getToolResultMessage(toolResponse.result);
  const jsonContent = formatToolResultAsJson(toolResponse.result);

  return (
    <>
      <AIChatBubble className={className}>
        <button
          type="button"
          onClick={() => setIsDialogOpen(true)}
          className="cursor-pointer rounded transition-colors hover:bg-neutral-100"
        >
          {content}
        </button>
      </AIChatBubble>

      <Dialog
        title={
          <span className="flex w-full items-center justify-between gap-4">
            <span className="flex items-center gap-2">
              <IconComponent
                size={18}
                weight="regular"
                className="text-neutral-500"
              />
              <span className="text-sm font-medium text-neutral-700">
                {getToolActionPhrase(toolResponse.toolName)}
              </span>
            </span>
            <button
              type="button"
              onClick={() => setShowDebug(!showDebug)}
              className={cn(
                "rounded px-2 py-1 text-xs transition-colors",
                showDebug
                  ? "bg-neutral-200 text-neutral-700"
                  : "bg-neutral-100 text-neutral-500 hover:bg-neutral-200",
              )}
            >
              {showDebug ? "Hide Debug" : "Debug"}
            </button>
          </span>
        }
        controlled={{
          isOpen: isDialogOpen,
          set: setIsDialogOpen,
        }}
        onClose={() => {
          setIsDialogOpen(false);
          setShowDebug(false);
        }}
        styling={{ maxWidth: 600, width: "100%", minWidth: "auto" }}
      >
        <Dialog.Content>
          <div className="max-h-[60vh] overflow-y-auto text-left text-[1rem] leading-relaxed">
            {showDebug ? (
              <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded-md bg-neutral-50 p-3 font-mono text-xs text-neutral-700">
                {jsonContent}
              </pre>
            ) : (
              <MarkdownContent content={messageContent} />
            )}
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
