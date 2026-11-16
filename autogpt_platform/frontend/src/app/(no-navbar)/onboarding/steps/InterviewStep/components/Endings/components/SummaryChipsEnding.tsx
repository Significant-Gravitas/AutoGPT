"use client";

import { HIDDEN_FROM_REPLAY } from "@/lib/replay-privacy";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { chipCorrections } from "../helpers";

interface Props {
  chips: string[];
  isLeaving: boolean;
  onConfirm: (corrections: string | null) => void;
}

// "Here's what I heard": tap a chip that is wrong to strike it, and add
// anything missed.
export function SummaryChipsEnding({ chips, isLeaving, onConfirm }: Props) {
  const [struck, setStruck] = useState<string[]>([]);
  const [added, setAdded] = useState("");

  function toggle(chip: string) {
    setStruck((current) =>
      current.includes(chip)
        ? current.filter((item) => item !== chip)
        : [...current, chip],
    );
  }

  return (
    <section
      aria-label="What Otto heard"
      className={cn(
        HIDDEN_FROM_REPLAY,
        "flex w-full max-w-md flex-col gap-4 rounded-large border border-border-base bg-surface p-5",
      )}
    >
      <div className="flex flex-col gap-1">
        <Text variant="large-medium">Here&apos;s what I heard</Text>
        <Text variant="small" tone="muted">
          Tap anything that&apos;s not right.
        </Text>
      </div>
      <ul className="flex flex-wrap gap-2">
        {chips.map((chip) => {
          const isStruck = struck.includes(chip);
          return (
            <li key={chip}>
              <button
                type="button"
                aria-pressed={isStruck}
                disabled={isLeaving}
                onClick={() => toggle(chip)}
                className={cn(
                  "rounded-full border px-3 py-1.5 text-left text-body transition-colors",
                  isStruck
                    ? "border-border-base bg-canvas text-foreground-tertiary line-through"
                    : "border-border-base bg-surface-hover text-foreground-base",
                )}
              >
                {chip}
              </button>
            </li>
          );
        })}
      </ul>
      <Input
        id="summary-add"
        label="Add something"
        size="small"
        placeholder="Anything Otto missed"
        value={added}
        maxLength={300}
        disabled={isLeaving}
        onChange={(event) => setAdded(event.target.value)}
      />
      <Button
        variant="primary"
        size="large"
        loading={isLeaving}
        disabled={isLeaving}
        onClick={() => onConfirm(chipCorrections(struck, added))}
      >
        Build my team
      </Button>
    </section>
  );
}
