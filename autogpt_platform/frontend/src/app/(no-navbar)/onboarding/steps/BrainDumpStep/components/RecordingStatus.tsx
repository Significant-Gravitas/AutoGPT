"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import { Text } from "@/components/atoms/Text/Text";
import { CloudOffIcon } from "@hugeicons/core-free-icons";
import { recordingFeedbackAt, SILENCE_NUDGE_COPY } from "../helpers";

interface Props {
  elapsedSeconds: number;
  showSilenceNudge: boolean;
  isOffline: boolean;
}

const OFFLINE_COPY = "You're offline — we'll send this when you're back.";

export function RecordingStatus({
  elapsedSeconds,
  showSilenceNudge,
  isOffline,
}: Props) {
  const feedback = showSilenceNudge
    ? null
    : recordingFeedbackAt(elapsedSeconds);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="mt-2 h-6">
        <SwapFade swapKey={feedback ?? "idle"} className="flex justify-center">
          <Text
            variant="small"
            className="text-center !text-sm !font-medium !text-purple-400"
          >
            {feedback ?? ""}
          </Text>
        </SwapFade>
      </div>

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
    </div>
  );
}
