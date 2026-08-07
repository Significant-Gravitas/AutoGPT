"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  canRecord: boolean;
  onRecordAgain: () => void;
  onTypeInstead: () => void;
  onSkip: () => void;
}

// The take went through — it just didn't say enough to personalize from.
// Calm and specific on purpose: nothing broke, nothing was lost, and all
// three ways forward stay open.
export function InsufficientState({
  canRecord,
  onRecordAgain,
  onTypeInstead,
  onSkip,
}: Props) {
  return (
    <div className="flex max-w-md flex-col items-center gap-4 text-center">
      <Text variant="lead" className="!text-base !text-zinc-500">
        We heard you, but there wasn&apos;t enough about your work to
        personalize things yet. A few sentences about what you do — in any
        language — is all it takes.
      </Text>
      <div className="flex flex-wrap items-center justify-center gap-3">
        {canRecord && (
          <Button variant="primary" size="small" onClick={onRecordAgain}>
            Record again
          </Button>
        )}
        <Button
          variant={canRecord ? "secondary" : "primary"}
          size="small"
          onClick={onTypeInstead}
        >
          Type instead
        </Button>
        <Button variant="ghost" size="small" onClick={onSkip}>
          Continue without it
        </Button>
      </div>
    </div>
  );
}
