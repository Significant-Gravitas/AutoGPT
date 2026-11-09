"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";
import { ReasonTiles, type Reason } from "./components/ReasonTiles";
import { ThanksLine } from "./components/ThanksLine";

interface Props {
  question: string;
  reasons: Reason[];
  onReason?: (value: string) => void;
  onContinue?: () => void;
}

export function OnboardingSkipPrompt({
  question,
  reasons,
  onReason,
  onContinue,
}: Props) {
  const [reason, setReason] = useState<string | null>(null);

  function pickReason(value: string) {
    setReason(value);
    onReason?.(value);
  }

  if (reason)
    return <ThanksLine withChangelog={false}>Got it, thanks.</ThanksLine>;

  return (
    <div className="flex w-full max-w-2xl flex-col items-center gap-3">
      <Text variant="small" tone="muted">
        {question}
      </Text>
      <div className="w-full">
        <ReasonTiles
          reasons={reasons}
          label={question}
          selected={reason}
          onSelect={pickReason}
          columns={reasons.length === 4 ? 4 : 3}
          layout="stack"
          size="compact"
        />
      </div>
      {onContinue ? (
        <Button variant="ghost" size="small" onClick={onContinue}>
          Continue
        </Button>
      ) : null}
    </div>
  );
}
