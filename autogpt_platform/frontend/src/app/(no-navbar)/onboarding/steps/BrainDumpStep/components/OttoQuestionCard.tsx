"use client";

import { Text } from "@/components/atoms/Text/Text";
import { OFF_YOUR_PLATE_QUESTION } from "./insufficient-copy";

// Otto's question from the not-enough screen, kept in view while the person
// answers it out loud.
export function OttoQuestionCard() {
  return (
    <div
      data-testid="otto-question-card"
      className="w-full max-w-xl rounded-3xl border border-brand/15 bg-accent-surface px-4 pt-3 pb-3.5 text-left md:px-5"
    >
      <Text variant="eyebrow" tone="accent" as="span">
        Otto
      </Text>
      <Text variant="h4" as="p" className="text-balance">
        {OFF_YOUR_PLATE_QUESTION}
      </Text>
    </div>
  );
}
