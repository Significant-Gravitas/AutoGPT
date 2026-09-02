"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  mode: "voice" | "typed";
  canRecord: boolean;
  onRecordAgain: () => void;
  onTypeInstead: () => void;
  onSkip: () => void;
}

// The take went through — it just didn't say enough to personalize from.
// Calm and specific on purpose: nothing broke, nothing was lost, and all
// three ways forward stay open. The copy and the primary action follow
// how the user got here: a voice reject leads with a fresh take, a typed
// reject leads back to their preserved text.
export function InsufficientState({
  mode,
  canRecord,
  onRecordAgain,
  onTypeInstead,
  onSkip,
}: Props) {
  const recordIsPrimary = mode === "voice" && canRecord;
  return (
    <div
      role="alert"
      className="flex max-w-md flex-col items-center gap-4 text-center"
    >
      <Text variant="lead" className="!text-base !text-zinc-500">
        {mode === "voice"
          ? "We heard you, but there wasn't enough about your work to " +
            "personalize things yet. A few sentences about what you do — " +
            "in any language — is all it takes."
          : "There wasn't enough about your work to personalize things " +
            "yet. A few sentences about what you do — in any language — " +
            "is all it takes."}
      </Text>
      <div className="flex flex-wrap items-center justify-center gap-3">
        {canRecord && (
          <Button
            variant={recordIsPrimary ? "primary" : "secondary"}
            size="small"
            onClick={onRecordAgain}
          >
            Record again
          </Button>
        )}
        <Button
          variant={recordIsPrimary ? "secondary" : "primary"}
          size="small"
          onClick={onTypeInstead}
        >
          {mode === "typed" ? "Add more detail" : "Type instead"}
        </Button>
        <Button variant="ghost" size="small" onClick={onSkip}>
          Continue without it
        </Button>
      </div>
    </div>
  );
}
