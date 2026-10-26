"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { formatElapsed } from "../helpers";

interface Props {
  durationSecs: number;
  onResume: () => void;
  onDownload: () => void;
  onDiscard: () => void;
}

// Reached when the tab was closed, refreshed or crashed mid-recording, or on
// Back after skipping past a recording. Everything captured is still in
// IndexedDB, so the only question is whether the user wants it.
export function RecoveryPrompt({
  durationSecs,
  onResume,
  onDownload,
  onDiscard,
}: Props) {
  return (
    <div className="flex w-full max-w-md flex-col items-center gap-4 text-center">
      <Text variant="h5">Pick up where you left off?</Text>
      <Text variant="body" tone="muted">
        We kept the{" "}
        <span className="text-foreground-strong">
          {formatElapsed(durationSecs)}
        </span>{" "}
        you already recorded.
      </Text>
      <div className="flex w-full flex-col items-center gap-2">
        <Button size="small" onClick={onResume} className="w-full max-w-xs">
          Use that recording
        </Button>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="xs" onClick={onDownload}>
            Download recording
          </Button>
          <Button variant="ghost" size="xs" onClick={onDiscard}>
            Start over
          </Button>
        </div>
      </div>
    </div>
  );
}
