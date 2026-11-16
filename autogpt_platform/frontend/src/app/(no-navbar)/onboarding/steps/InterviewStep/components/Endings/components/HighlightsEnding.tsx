"use client";

import type { Fact } from "@/app/api/__generated__/models/fact";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useState } from "react";
import {
  correctionsOf,
  highlightSegments,
  notRightCorrection,
} from "../helpers";

interface Props {
  words: string[];
  facts: Fact[];
  isLeaving: boolean;
  onConfirm: (corrections: string | null) => void;
}

// The person's own words, with what Otto took from them underlined; tap an
// underline that is wrong to strike it.
export function HighlightsEnding({
  words,
  facts,
  isLeaving,
  onConfirm,
}: Props) {
  const [struck, setStruck] = useState<number[]>([]);
  const quotes = facts.map((fact) => fact.quote);

  function toggle(fact: number) {
    setStruck((current) =>
      current.includes(fact)
        ? current.filter((item) => item !== fact)
        : [...current, fact],
    );
  }

  function handleConfirm() {
    onConfirm(
      correctionsOf(
        struck.map((fact) => notRightCorrection(facts[fact]?.label ?? "")),
      ),
    );
  }

  return (
    <section
      aria-label="In your words"
      className="flex w-full max-w-md flex-col gap-4 rounded-large border border-border-base bg-surface p-5"
    >
      <div className="flex flex-col gap-1">
        <Text variant="large-medium">In your words</Text>
        <Text variant="small" tone="muted">
          Underlined is what Otto took from it. Tap anything that&apos;s wrong.
        </Text>
      </div>
      <div className="flex max-h-72 flex-col gap-2 overflow-y-auto">
        {words.map((line, index) => (
          <Text key={`${index}-${line.slice(0, 12)}`} variant="body">
            {highlightSegments(line, quotes).map((segment, at) =>
              segment.fact === null ? (
                <span key={at}>{segment.text}</span>
              ) : (
                <button
                  key={at}
                  type="button"
                  aria-pressed={struck.includes(segment.fact)}
                  aria-label={`${facts[segment.fact]?.label}: ${segment.text}`}
                  disabled={isLeaving}
                  onClick={() => toggle(segment.fact as number)}
                  className={cn(
                    "underline decoration-2 underline-offset-4",
                    struck.includes(segment.fact)
                      ? "text-foreground-tertiary line-through decoration-border-base"
                      : "decoration-brand",
                  )}
                >
                  {segment.text}
                </button>
              ),
            )}
          </Text>
        ))}
      </div>
      <Button
        variant="primary"
        size="large"
        loading={isLeaving}
        disabled={isLeaving}
        onClick={handleConfirm}
      >
        Looks right
      </Button>
    </section>
  );
}
