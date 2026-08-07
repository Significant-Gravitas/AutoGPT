"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  onDownload: () => void;
  onSkip: () => void;
}

// Calm on purpose. The recording is safe in the browser and (once it got
// that far) on the server, so this is an inconvenience, not a loss — the
// copy and the download button both have to make that obvious. Retrying is
// the orb itself, so only the secondary exits live here.
export function FailureState({ onDownload, onSkip }: Props) {
  return (
    <div className="flex flex-col items-center gap-4 text-center">
      <Text variant="lead" className="!text-base !text-zinc-500">
        Your recording is safe. Try again.
      </Text>
      <div className="flex items-center gap-3">
        <Button variant="primary" size="small" onClick={onDownload}>
          Download recording
        </Button>
        <Button variant="ghost" size="small" onClick={onSkip}>
          Continue without it
        </Button>
      </div>
    </div>
  );
}
