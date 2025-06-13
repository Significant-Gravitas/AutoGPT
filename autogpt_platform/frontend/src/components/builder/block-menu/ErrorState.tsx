import { Button } from "@/components/ui/button";
import { cn, parseErrorMessage } from "@/lib/utils";
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
  return (
    <div
      className={cn(
        "flex h-full w-full flex-col items-center justify-center space-y-4 text-center",
        className,
      )}
    >
      {showIcon && <AlertCircle className="h-12 w-12" strokeWidth={1.5} />}

      <div className="space-y-2">
        <p className="text-sm font-medium text-zinc-800">{title}</p>
        <p className="text-sm text-zinc-600">
          {parseErrorMessage(error, message)}
        </p>
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
