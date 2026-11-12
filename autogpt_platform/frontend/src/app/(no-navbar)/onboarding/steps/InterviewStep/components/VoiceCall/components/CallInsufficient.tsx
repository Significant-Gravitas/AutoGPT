"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  onCallAgain: () => void;
  onTypeInstead: () => void;
  onSkip: () => void;
}

// The call went through; its words just didn't say enough to personalize
// from. Calm and specific on purpose: nothing broke, nothing was lost, and
// all three ways forward stay open, a fresh call first.
export function CallInsufficient({
  onCallAgain,
  onTypeInstead,
  onSkip,
}: Props) {
  return (
    <div
      role="alert"
      className="flex max-w-md flex-col items-center gap-4 text-center"
    >
      <Text variant="body" tone="muted">
        {"We heard you, but there wasn't enough about your work to " +
          "personalize things yet. A few sentences about what you do — " +
          "in any language — is all it takes."}
      </Text>
      <div className="flex flex-wrap items-center justify-center gap-3">
        <Button variant="primary" size="small" onClick={onCallAgain}>
          Call again
        </Button>
        <Button variant="secondary" size="small" onClick={onTypeInstead}>
          Type instead
        </Button>
        <Button variant="ghost" size="small" onClick={onSkip}>
          Continue without it
        </Button>
      </div>
    </div>
  );
}
