"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { useState } from "react";
import { ReasonFlow } from "./components/ReasonFlow";
import { WORK_REASONS } from "./helpers";

interface Props {
  initialSent?: boolean;
  onReason?: (value: string) => void;
  onSubmit?: (answer: { reason: string; note: string }) => void;
  onDismiss?: () => void;
}

export function OttoMistakeCard({
  initialSent = false,
  onReason,
  onSubmit,
  onDismiss,
}: Props) {
  const [isOpen, setIsOpen] = useState(true);

  function close() {
    setIsOpen(false);
    onDismiss?.();
  }

  if (!isOpen) return null;

  return (
    <div className="flex max-w-xl flex-col gap-4 rounded-2xl border border-border-base bg-surface p-4 shadow-sm">
      <div className="flex items-start gap-3">
        <AutopilotAvatar size={32} className="ring-1 ring-border-base" />
        <div className="flex flex-1 flex-col gap-0.5">
          <Text variant="body-medium" tone="primary">
            Help the team fix this
          </Text>
          <Text variant="small" tone="muted">
            I&apos;ve flagged this chat for review. Which of these went wrong?
          </Text>
        </div>
        <Button
          variant="ghost"
          size="xs"
          className="-mt-1 -mr-1"
          onClick={close}
        >
          Not now
        </Button>
      </div>
      <ReasonFlow
        reasons={WORK_REASONS}
        label="Which of these went wrong?"
        size="compact"
        thanks="Thanks. The team reads every one of these."
        initialSent={initialSent}
        onReason={onReason}
        onSubmit={onSubmit}
      />
    </div>
  );
}
