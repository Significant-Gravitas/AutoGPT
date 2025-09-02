"use client";

import React, { useState, useRef, useEffect, KeyboardEvent } from "react";
import { Send, X } from "lucide-react";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  onStopStreaming?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
  placeholder?: string;
  maxLength?: number;
  className?: string;
}

export function ChatInput({
  onSendMessage,
  onStopStreaming,
  isStreaming = false,
  disabled = false,
  placeholder = "Ask about AI agents or describe what you want to automate...",
  maxLength = 10000,
  className,
}: ChatInputProps) {
  const [message, setMessage] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [message]);

  const handleSubmit = () => {
    if (message.trim() && !disabled && !isStreaming) {
      onSendMessage(message.trim());
      setMessage("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const isDisabled = disabled || (isStreaming && !onStopStreaming);
  const charactersRemaining = maxLength - message.length;
  const showCharacterCount = message.length > maxLength * 0.8;

  return (
    <div className={cn("border-t border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900", className)}>
      <div className="mx-auto max-w-4xl px-4 py-4">
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={isDisabled}
              maxLength={maxLength}
              className={cn(
                "w-full resize-none rounded-lg border border-neutral-300 dark:border-neutral-600",
                "bg-white dark:bg-neutral-800 px-4 py-3",
                "text-neutral-900 dark:text-neutral-100",
                "placeholder:text-neutral-500 dark:placeholder:text-neutral-400",
                "focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-500/20",
                "disabled:cursor-not-allowed disabled:opacity-50",
                "min-h-[52px] max-h-[200px]"
              )}
              rows={1}
            />
            {showCharacterCount && (
              <div
                className={cn(
                  "mt-1 text-xs",
                  charactersRemaining < 100
                    ? "text-red-500"
                    : "text-neutral-500 dark:text-neutral-400"
                )}
              >
                {charactersRemaining} characters remaining
              </div>
            )}
          </div>
          
          {isStreaming && onStopStreaming ? (
            <Button
              onClick={onStopStreaming}
              variant="secondary"
              size="md"
              className="mb-[2px]"
            >
              <X className="mr-1 h-4 w-4" />
              Stop
            </Button>
          ) : (
            <Button
              onClick={handleSubmit}
              disabled={!message.trim() || isDisabled}
              variant="primary"
              size="md"
              className="mb-[2px]"
            >
              <Send className="mr-1 h-4 w-4" />
              Send
            </Button>
          )}
        </div>
        
        <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  );
}