"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useIsNarrowScreen } from "../useIsNarrowScreen";
import { OFF_YOUR_PLATE_QUESTION } from "./insufficient-copy";
import { TypingDots } from "./TypingDots";
import { useOttoTyping } from "./useOttoTyping";

interface Props {
  firstLine: string;
}

// Otto's side of the chat: a bubble whose tail points up at him. His words
// are in the DOM from the first render, so a screen reader reads them at
// once; only their opacity waits for the typing dots.
export function OttoReply({ firstLine }: Props) {
  const isNarrow = useIsNarrowScreen();
  const { isTyping, reduceMotion } = useOttoTyping();

  return (
    <div
      role="alert"
      className="relative w-full rounded-3xl border border-brand/15 bg-accent-surface px-5 pt-4 pb-5 text-left"
    >
      <span
        aria-hidden="true"
        className="absolute -top-2 left-1/2 size-4 -translate-x-1/2 rotate-45 rounded-tl-sm border-t border-l border-brand/15 bg-accent-surface"
      />
      <Text variant="eyebrow" tone="accent" as="span">
        Otto
      </Text>
      <div className="relative">
        {isTyping && <TypingDots />}
        <div
          data-testid="otto-reply-message"
          className={cn(
            !reduceMotion && "transition-opacity duration-300",
            isTyping && "opacity-0",
          )}
        >
          <Text variant="large" tone="body" className="mt-1.5">
            {firstLine}
          </Text>
          <Text
            variant={isNarrow ? "h3" : "h2"}
            as="h2"
            className="mt-0.5 text-balance"
          >
            {OFF_YOUR_PLATE_QUESTION}
          </Text>
        </div>
      </div>
    </div>
  );
}
