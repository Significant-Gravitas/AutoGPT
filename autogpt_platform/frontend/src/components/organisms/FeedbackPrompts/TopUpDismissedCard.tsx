"use client";

import { useState } from "react";
import { PromptCard } from "./components/PromptCard";
import { ReasonFlow } from "./components/ReasonFlow";
import { TOP_UP_REASONS } from "./helpers";

interface Props {
  onReason?: (value: string) => void;
  onSubmit?: (answer: { reason: string; note: string }) => void;
  onDismiss?: () => void;
  onClose?: () => void;
}

export function TopUpDismissedCard({
  onReason,
  onSubmit,
  onDismiss,
  onClose,
}: Props) {
  const [isOpen, setIsOpen] = useState(true);
  const [isSent, setIsSent] = useState(false);

  function send(answer: { reason: string; note: string }) {
    setIsSent(true);
    onSubmit?.(answer);
  }

  function close() {
    setIsOpen(false);
    if (!isSent) onDismiss?.();
    onClose?.();
  }

  if (!isOpen) return null;

  return (
    <PromptCard
      eyebrow="Quick question"
      title="What would make topping up worth it?"
      onClose={close}
    >
      <ReasonFlow
        reasons={TOP_UP_REASONS}
        label="What would make topping up worth it?"
        size="compact"
        onReason={onReason}
        onSubmit={send}
      />
    </PromptCard>
  );
}
