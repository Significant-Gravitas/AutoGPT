import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { AlertCircle, RefreshCw } from "lucide-react";
import React from "react";

interface ErrorStateProps {
  title?: string;
  message?: string;
  error?: string | Error | null;
  onRetry?: () => void;
  retryLabel?: string;
  className?: string;
  showIcon?: boolean;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = "Something went wrong",
  message,
  error,
  onRetry,
  retryLabel = "Retry",
  className,
  showIcon = true,
}) => {
  const errorMessage = error
    ? error instanceof Error
      ? error.message
      : String(error)
    : message || "An unexpected error occurred. Please try again.";

  const classes =
    "flex h-full w-full flex-col items-center justify-center text-center space-y-4";

  return (
    <div className={cn(classes, className)}>
      {showIcon && <AlertCircle className="h-12 w-12" strokeWidth={1.5} />}

      <div className="space-y-2">
        <p className="text-sm font-medium text-zinc-800">{title}</p>
        <p className="text-sm text-zinc-600">{errorMessage}</p>
      </div>

      {onRetry && (
        <Button
          variant="default"
          size="sm"
          onClick={onRetry}
          className="mt-2 h-7 bg-zinc-800 text-xs"
        >
          <RefreshCw className="mr-1 h-3 w-3" />
          {retryLabel}
        </Button>
      )}
    </div>
  );
};


