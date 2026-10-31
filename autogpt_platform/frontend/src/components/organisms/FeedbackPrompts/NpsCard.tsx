"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { PromptCard } from "./components/PromptCard";
import { ThanksLine } from "./components/ThanksLine";

const SCORES = Array.from({ length: 11 }, (_, i) => i);

interface Props {
  initialScore?: number | null;
  initialSent?: boolean;
}

function followUpFor(score: number) {
  if (score >= 9) return "What do you like most?";
  if (score >= 7) return "What would make it a 10?";
  return "What's the biggest thing holding it back?";
}

export function NpsCard({ initialScore = null, initialSent = false }: Props) {
  const [score, setScore] = useState<number | null>(initialScore);
  const [note, setNote] = useState("");
  const [sent, setSent] = useState(initialSent);
  const [isOpen, setIsOpen] = useState(true);

  if (!isOpen) return null;

  return (
    <PromptCard
      title="How likely are you to recommend AutoGPT?"
      onClose={() => setIsOpen(false)}
    >
      {sent ? (
        <ThanksLine />
      ) : (
        <div className="flex flex-col gap-3">
          <div
            role="group"
            aria-label="How likely are you to recommend AutoGPT, from 0 (not likely) to 10 (very likely)"
            className="flex gap-1"
          >
            {SCORES.map((value) => (
              <button
                key={value}
                type="button"
                aria-pressed={score === value}
                onClick={() => setScore(value)}
                className={cn(
                  "h-8 flex-1 rounded-md border border-border-base bg-surface text-caption font-medium text-foreground-emphasis transition-colors hover:border-border-strong hover:bg-surface-muted",
                  "aria-pressed:border-action aria-pressed:bg-action aria-pressed:text-action-foreground",
                )}
              >
                {value}
              </button>
            ))}
          </div>
          <div className="flex justify-between">
            <Text variant="small" className="!text-foreground-tertiary">
              Not likely
            </Text>
            <Text variant="small" className="!text-foreground-tertiary">
              Very likely
            </Text>
          </div>
          {score !== null ? (
            <>
              <Text variant="body-medium" className="!text-foreground-strong">
                {followUpFor(score)}
              </Text>
              <Textarea
                rows={3}
                value={note}
                onChange={(e) => setNote(e.target.value)}
                aria-label={followUpFor(score)}
                placeholder="Optional"
                className="resize-none"
              />
              <div className="flex justify-end">
                <Button size="small" onClick={() => setSent(true)}>
                  Send
                </Button>
              </div>
            </>
          ) : null}
        </div>
      )}
    </PromptCard>
  );
}
