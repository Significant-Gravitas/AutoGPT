"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { formatElapsed } from "../helpers";

interface Props {
  durationSecs: number;
  onResume: () => void;
  onDiscard: () => void;
}

// Reached when the tab was closed, refreshed or crashed mid-recording.
// Everything captured is still in IndexedDB, so the only question is
// whether the user wants it.
export function RecoveryPrompt({ durationSecs, onResume, onDiscard }: Props) {
  return (
    <div className="flex w-full max-w-md flex-col items-center gap-6 text-center">
      <Text
        variant="h3"
        className="!text-[1.125rem] !leading-[1.625rem] md:!text-[1.25rem] md:!leading-[1.75rem]"
      >
        Pick up where you left off?
      </Text>
      <Text variant="lead" className="!text-base !text-zinc-500">
        We kept the{" "}
        <span className="text-purple-500">{formatElapsed(durationSecs)}</span>{" "}
        you already recorded.
      </Text>
      <div className="flex w-full flex-col items-center gap-3">
        <Button onClick={onResume} className="w-full max-w-xs">
          Use that recording
        </Button>
        <button
          type="button"
          onClick={onDiscard}
          className="text-sm text-zinc-700 transition-colors hover:text-zinc-900"
        >
          Start over
        </button>
      </div>
    </div>
  );
}
