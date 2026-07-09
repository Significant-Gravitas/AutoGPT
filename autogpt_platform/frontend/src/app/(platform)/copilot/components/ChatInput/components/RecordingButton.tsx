"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { CircleNotchIcon } from "@/components/atoms/AGPTIcon/icons";
import { MicIcon } from "@/components/icons/MicIcon";

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
        "!size-9 border-0 bg-transparent !p-0 text-zinc-500 hover:border-0 hover:bg-zinc-100 hover:text-zinc-700 focus-visible:ring-0",
        disabled && "opacity-40",
        isRecording && "animate-pulse bg-red-500 text-white hover:bg-red-600",
        isTranscribing && "bg-zinc-100 text-zinc-400",
        isStreaming && "opacity-40",
      )}
    >
      {isTranscribing ? (
        <CircleNotchIcon className="h-4 w-4 animate-spin" weight="bold" />
      ) : (
        <MicIcon className="h-4 w-4" />
      )}
    </Button>
  );
}
