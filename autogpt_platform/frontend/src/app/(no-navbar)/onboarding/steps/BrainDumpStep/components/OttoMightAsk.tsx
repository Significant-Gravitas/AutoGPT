"use client";

import { Button } from "@/components/atoms/Button/Button";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  question: string;
  // Without it there is no "Another question": the phone's rest screen
  // shows the prompt as a hint only.
  onAnother?: () => void;
  // The question the user just asked for, read out once; empty otherwise.
  announcement?: string;
  align?: "left" | "center";
  compact?: boolean;
  // A click on the question pauses or resumes every idea that changes on its
  // own; there is no visible control for it, on purpose.
  onTogglePause?: () => void;
  className?: string;
}

export function OttoMightAsk({
  question,
  onAnother,
  announcement = "",
  align = "center",
  compact = false,
  onTogglePause,
  className,
}: Props) {
  const isLeft = align === "left";

  // Left aligned above the composer, the button shares the label's row to
  // save the composer a line; the DOM keeps label, question, button order
  // for screen readers.
  return (
    <div
      className={cn(
        "w-full",
        isLeft
          ? "grid grid-cols-[1fr_auto] items-center gap-x-3 text-left"
          : "flex flex-col items-center text-center",
        className,
      )}
    >
      <Text
        variant="small-semibold"
        as="span"
        className={cn(
          "text-caption-sm uppercase tracking-eyebrow text-accent-strong",
          isLeft && "col-start-1 row-start-1",
        )}
      >
        Otto might ask
      </Text>
      <SwapFade
        swapKey={question}
        className={cn("mt-0.5", isLeft && "col-span-2 row-start-2")}
      >
        <Text
          variant={compact ? "body" : "large"}
          tone={compact ? "secondary" : undefined}
          className={cn("text-pretty", onTogglePause && "cursor-pointer")}
          onClick={onTogglePause}
        >
          {question}
        </Text>
      </SwapFade>
      {onAnother && (
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={onAnother}
          className={cn(
            "underline underline-offset-4",
            isLeft ? "col-start-2 row-start-1 -mr-2.5" : "mt-1",
          )}
        >
          Another question
        </Button>
      )}
      {/* The only announcement of the prompt layer: the ambient cards are
          hidden from assistive tech, and a question the timer swaps in is
          left for the user to find. */}
      <p
        aria-live="polite"
        data-testid="otto-might-ask-announcer"
        className="sr-only"
      >
        {announcement && `Otto might ask: ${announcement}`}
      </p>
    </div>
  );
}
