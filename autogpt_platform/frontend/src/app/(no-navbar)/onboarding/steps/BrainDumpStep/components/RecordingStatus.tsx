"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { encouragementAt, SILENCE_NUDGE_COPY } from "../helpers";

interface Props {
  elapsedSeconds: number;
  showSilenceNudge: boolean;
}

export function RecordingStatus({ elapsedSeconds, showSilenceNudge }: Props) {
  const encouragement = encouragementAt(elapsedSeconds);

  return (
    <div className="flex flex-col items-center gap-2">
      <Text
        variant="small"
        className={cn(
          "h-5 text-center transition-opacity duration-500",
          encouragement ? "!text-purple-600 opacity-100" : "opacity-0",
        )}
      >
        {encouragement ?? ""}
      </Text>

      {showSilenceNudge && (
        <Text variant="small" className="max-w-sm text-center !text-zinc-500">
          {SILENCE_NUDGE_COPY}
        </Text>
      )}
    </div>
  );
}
