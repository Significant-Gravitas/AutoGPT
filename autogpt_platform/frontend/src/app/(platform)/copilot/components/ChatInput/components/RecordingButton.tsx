"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { CircleNotchIcon, MicrophoneIcon } from "@phosphor-icons/react";

interface Props {
  isRecording: boolean;
  isTranscribing: boolean;
  isStreaming: boolean;
  disabled: boolean;
  onClick: () => void;
}

export function RecordingButton({
  isRecording,
  isTranscribing,
  isStreaming,
  disabled,
  onClick,
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
        "border-0 bg-white text-zinc-500 hover:bg-zinc-50 hover:text-zinc-700",
        disabled && "opacity-40",
        isRecording && "animate-pulse bg-red-500 text-white hover:bg-red-600",
        isTranscribing && "bg-zinc-100 text-zinc-400",
        isStreaming && "opacity-40",
      )}
    >
      {isTranscribing ? (
        <CircleNotchIcon className="h-4 w-4 animate-spin" weight="bold" />
      ) : (
        <MicrophoneIcon className="h-4 w-4" weight="bold" />
      )}
    </Button>
  );
}
