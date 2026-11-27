"use client";

import { ReasonFlow } from "./components/ReasonFlow";
import { TOP_UP_REASONS } from "./helpers";

export const TOP_UP_QUESTION = "What would make topping up worth it?";

interface Props {
  onReason?: (value: string) => void;
  onSubmit?: (answer: { reason: string; note: string }) => void;
}

// Asked inside the out-of-credits dialog, after "Not now".
export function TopUpQuestion({ onReason, onSubmit }: Props) {
  return (
    <ReasonFlow
      reasons={TOP_UP_REASONS}
      label={TOP_UP_QUESTION}
      size="compact"
      onReason={onReason}
      onSubmit={onSubmit}
    />
  );
}
