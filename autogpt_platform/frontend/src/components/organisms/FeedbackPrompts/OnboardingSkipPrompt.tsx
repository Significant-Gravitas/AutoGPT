"use client";

import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";
import { ReasonTiles, type Reason } from "./components/ReasonTiles";
import { ThanksLine } from "./components/ThanksLine";

interface Props {
  question: string;
  reasons: Reason[];
}

export function OnboardingSkipPrompt({ question, reasons }: Props) {
  const [reason, setReason] = useState<string | null>(null);

  if (reason)
    return <ThanksLine withChangelog={false}>Got it, thanks.</ThanksLine>;

  return (
    <div className="flex w-full max-w-2xl flex-col items-center gap-3">
      <Text variant="small" className="!text-foreground-tertiary">
        {question}
      </Text>
      <div className="w-full">
        <ReasonTiles
          reasons={reasons}
          label={question}
          selected={reason}
          onSelect={setReason}
          columns={reasons.length === 4 ? 4 : 3}
          layout="stack"
          size="compact"
        />
      </div>
    </div>
  );
}
