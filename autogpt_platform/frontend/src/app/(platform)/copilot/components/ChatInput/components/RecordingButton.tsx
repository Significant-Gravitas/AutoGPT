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
  // dump: AutoPilot's intro invites them to record, so the button it
  // points at has to be findable.
  highlight?: boolean;
}

export function RecordingButton({
  isRecording,
  isTranscribing,
  isStreaming,
  disabled,
  onClick,
  highlight = false,
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
        "border-neutral-200 bg-white text-zinc-500 shadow-sm hover:border-neutral-200 hover:bg-neutral-50 hover:text-zinc-700",
        disabled && "opacity-40",
        isRecording && "animate-pulse bg-red-500 text-white hover:bg-red-600",
        isTranscribing && "bg-zinc-100 text-zinc-400",
        isStreaming && "opacity-40",
        highlight &&
          !isRecording &&
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
