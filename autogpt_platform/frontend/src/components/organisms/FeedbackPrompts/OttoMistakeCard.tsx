"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { useState } from "react";
import { ReasonFlow } from "./components/ReasonFlow";
import { MISTAKE_REASONS } from "./helpers";

export function OttoMistakeCard() {
  const [isOpen, setIsOpen] = useState(true);

  if (!isOpen) return null;

  return (
    <div className="flex max-w-xl flex-col gap-4 rounded-2xl border border-border-base bg-surface p-4 shadow-sm">
      <div className="flex items-start gap-3">
        <AutopilotAvatar size={32} className="ring-1 ring-border-base" />
        <div className="flex flex-1 flex-col gap-0.5">
          <Text variant="body-medium" className="!text-foreground-strong">
            Help the team fix this
          </Text>
          <Text variant="small" className="!text-foreground-tertiary">
            I&apos;ve flagged this chat for review. Which of these went wrong?
          </Text>
        </div>
        <Button
          variant="ghost"
          size="xs"
          className="-mr-1 -mt-1"
          onClick={() => setIsOpen(false)}
        >
          Not now
        </Button>
      </div>
      <ReasonFlow
        reasons={MISTAKE_REASONS}
        size="compact"
        thanks="Thanks. The team reads every one of these."
      />
    </div>
  );
}
