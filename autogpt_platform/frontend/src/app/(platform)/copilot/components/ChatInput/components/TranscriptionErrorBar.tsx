"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Cancel01Icon,
  Download04Icon,
  Mic01Icon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";

interface Props {
  message: string;
  isRetrying: boolean;
  /** Absent when the same audio cannot work a second time. */
  onRetry?: () => void;
  /** Offered in place of Retry: a new recording is the only way on. */
  onRecordAgain?: () => void;
  onDownload: () => void;
  onDismiss: () => void;
  className?: string;
}

/**
 * The recording behind this message is still in memory, so the row stays put
 * until the user acts on it — retried, recorded again, downloaded, or
 * dismissed. Anything
 * that expires on its own takes the audio with it.
 */
export function TranscriptionErrorBar({
  message,
  isRetrying,
  onRetry,
  onRecordAgain,
  onDownload,
  onDismiss,
  className,
}: Props) {
  return (
    <div
      role="alert"
      className={cn(
        "flex w-full flex-wrap items-center gap-1.5 rounded-2xl bg-error-surface px-3 py-1.5 text-body text-error-foreground",
        className,
      )}
    >
      <Icon icon={Alert02Icon} className="h-4 w-4 shrink-0" aria-hidden />
      <span className="min-w-0 flex-1 truncate">{message}</span>
      {onRetry && (
        <Button
          type="button"
          variant="ghost"
          size="xs"
          leadingIcon={RefreshIcon}
          loading={isRetrying}
          disabled={isRetrying}
          onClick={onRetry}
        >
          Retry
        </Button>
      )}
      {onRecordAgain && (
        <Button
          type="button"
          variant="ghost"
          size="xs"
          leadingIcon={Mic01Icon}
          onClick={onRecordAgain}
        >
          Record again
        </Button>
      )}
      <Button
        type="button"
        variant="ghost"
        size="xs"
        leadingIcon={Download04Icon}
        onClick={onDownload}
      >
        Download recording
      </Button>
      <Button
        type="button"
        variant="icon"
        size="icon-xs"
        aria-label="Dismiss"
        onClick={onDismiss}
      >
        <Icon icon={Cancel01Icon} className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
