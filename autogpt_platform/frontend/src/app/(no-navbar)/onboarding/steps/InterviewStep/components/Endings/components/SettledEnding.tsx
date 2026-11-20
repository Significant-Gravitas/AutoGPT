"use client";

import { HIDDEN_FROM_REPLAY } from "@/lib/replay-privacy";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { SettledView } from "../helpers";

interface Props {
  view: SettledView;
  isLeaving: boolean;
  onContinue: () => void;
}

// The profile card once the interview is handed off: read-only, and the
// way on.
export function SettledEnding({ view, isLeaving, onContinue }: Props) {
  return (
    <section
      aria-label={view.heading}
      className={cn(
        HIDDEN_FROM_REPLAY,
        "flex w-full max-w-md flex-col gap-4 rounded-large border border-border-base bg-surface p-5",
      )}
    >
      <Text variant="large-medium">{view.heading}</Text>
      <dl className="flex flex-col gap-3">
        {view.rows.map((row) => (
          <div key={row.label} className="flex flex-col gap-1">
            <dt>
              <Text variant="small" tone="muted">
                {row.label}
              </Text>
            </dt>
            <dd>
              <Text variant="body-medium">{row.value}</Text>
            </dd>
          </div>
        ))}
      </dl>
      <Button
        variant="primary"
        size="large"
        loading={isLeaving}
        disabled={isLeaving}
        onClick={onContinue}
      >
        Continue
      </Button>
    </section>
  );
}
