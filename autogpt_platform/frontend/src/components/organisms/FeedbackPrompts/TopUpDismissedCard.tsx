"use client";

import { useState } from "react";
import { PromptCard } from "./components/PromptCard";
import { ReasonFlow } from "./components/ReasonFlow";
import { TOP_UP_REASONS } from "./helpers";

export function TopUpDismissedCard() {
  const [isOpen, setIsOpen] = useState(true);

  if (!isOpen) return null;

  return (
    <PromptCard
      eyebrow="Quick question"
      title="What would make topping up worth it?"
      onClose={() => setIsOpen(false)}
    >
      <ReasonFlow
        reasons={TOP_UP_REASONS}
        label="What would make topping up worth it?"
        size="compact"
      />
    </PromptCard>
  );
}
