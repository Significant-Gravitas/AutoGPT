import React from "react";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";

export interface ChatErrorStateProps {
  error: Error;
  onRetry?: () => void;
  className?: string;
}

export function ChatErrorState({
  error,
  onRetry,
  className,
}: ChatErrorStateProps) {
  return (
    <div
      className={cn("flex flex-1 items-center justify-center p-6", className)}
    >
      <ErrorCard
        responseError={{
          message: error.message,
        }}
        context="chat session"
        onRetry={onRetry}
        className="max-w-md"
      />
    </div>
  );
}
