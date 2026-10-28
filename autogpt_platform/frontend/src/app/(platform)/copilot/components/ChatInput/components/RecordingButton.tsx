"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { Loading03Icon, Mic01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  isRecording: boolean;
  isTranscribing: boolean;
  isStreaming: boolean;
  disabled: boolean;
  onClick: () => void;
  // One-time highlight for the user who skipped the onboarding brain
  // dump: Otto's intro invites them to record, so the button it
  // points at has to be findable.
  highlight?: boolean;
  className?: string;
}

export function RecordingButton({
  isRecording,
  isTranscribing,
  isStreaming,
  disabled,
  onClick,
  highlight = false,
  className,
}: Props) {
  return (
    <Button
      type="button"
      variant="icon"
      size="icon"
      aria-label={isRecording ? "Stop recording" : "Start recording"}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "border-transparent bg-transparent text-foreground-base shadow-none hover:border-transparent hover:bg-surface-hover hover:text-foreground-base",
        className,
        disabled && "opacity-40",
        isRecording &&
          "animate-pulse bg-error text-action-foreground hover:bg-error-hover",
        isTranscribing && "bg-surface-hover text-foreground-subtle",
        isStreaming && "opacity-40",
        highlight &&
          !isRecording &&
          // TODO(tokens): no exact role for border-purple-300, ring-purple-100; no fitting role for text-purple-600 #6C2EDF; #6C2EDF is brand-hover (a fill)
          "border-purple-300 text-purple-600 ring-4 ring-purple-100",
      )}
    >
      {isTranscribing ? (
        <Icon icon={Loading03Icon} className="h-4 w-4 animate-spin" />
      ) : (
        <Icon icon={Mic01Icon} className="h-4 w-4" />
      )}
    </Button>
  );
}
