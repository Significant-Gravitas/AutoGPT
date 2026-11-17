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

// A confirm arm's ending once the interview is handed off: what was
// confirmed, read-only, and the way on.
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
      {view.rows.length > 0 ? (
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
      ) : (
        <ul className="flex flex-wrap gap-2">
          {view.items.map((item) => (
            <li
              key={item}
              className="rounded-full border border-border-base bg-surface-hover px-3 py-1.5 text-body text-foreground-base"
            >
              {item}
            </li>
          ))}
        </ul>
      )}
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
