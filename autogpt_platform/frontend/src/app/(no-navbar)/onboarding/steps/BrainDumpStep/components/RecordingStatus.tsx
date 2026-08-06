"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { CloudOffIcon } from "@hugeicons/core-free-icons";
import { encouragementAt, SILENCE_NUDGE_COPY } from "../helpers";

interface Props {
  elapsedSeconds: number;
  showSilenceNudge: boolean;
  isOffline: boolean;
  isSavedLocally: boolean;
}

const OFFLINE_COPY = "You're offline — we'll send this when you're back.";
const SAVED_LOCALLY_COPY = "Saved on this device as you talk";

export function RecordingStatus({
  elapsedSeconds,
  showSilenceNudge,
  isOffline,
  isSavedLocally,
}: Props) {
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

      {/* A dropped connection is not a lost recording, and saying so is the
          whole point of writing every chunk to the device first. */}
      {isOffline && (
        <div className="flex items-center gap-2 rounded-full bg-zinc-100 px-3 py-1">
          <Icon
            icon={CloudOffIcon}
            size={14}
            className="shrink-0 text-zinc-500"
          />
          <Text variant="small" className="!text-zinc-500">
            {OFFLINE_COPY}
          </Text>
        </div>
      )}

      {isSavedLocally && (
        <Text variant="small" className="text-center !text-zinc-400">
          {SAVED_LOCALLY_COPY}
        </Text>
      )}
    </div>
  );
}
