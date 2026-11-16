"use client";

import { HIDDEN_FROM_REPLAY } from "@/lib/replay-privacy";
import { cn } from "@/lib/utils";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";
import { type ProfileRow, profileCorrections } from "../helpers";

interface Props {
  rows: ProfileRow[];
  // Live, it fills in while the person talks, and has no button yet.
  isLive?: boolean;
  isLeaving?: boolean;
  onConfirm?: (corrections: string | null) => void;
}

// The person as the interview heard them; each row can be put right.
export function ProfileCardEnding({
  rows,
  isLive = false,
  isLeaving = false,
  onConfirm,
}: Props) {
  const [edited, setEdited] = useState(rows);
  const [editing, setEditing] = useState<number | null>(null);
  const shown = isLive ? rows : edited;

  function handleChange(index: number, value: string) {
    setEdited((current) =>
      current.map((row, at) => (at === index ? { ...row, value } : row)),
    );
  }

  return (
    <section
      aria-label="Your profile"
      className={cn(
        HIDDEN_FROM_REPLAY,
        "flex w-full max-w-md flex-col gap-3 rounded-large border border-border-base bg-surface p-5",
      )}
    >
      <Text variant="small-medium" tone="muted">
        {isLive ? "What Otto has heard so far" : "Is this you?"}
      </Text>
      <dl className="flex flex-col gap-3">
        {shown.map((row, index) => (
          <div key={row.key} className="flex flex-col gap-1">
            <dt>
              <Text variant="small" tone="muted">
                {row.label}
              </Text>
            </dt>
            <dd className="flex items-start justify-between gap-2">
              {editing === index ? (
                <Input
                  id={`profile-${row.key}`}
                  label={row.label}
                  hideLabel
                  size="small"
                  value={row.value}
                  autoFocus
                  onChange={(event) => handleChange(index, event.target.value)}
                  onBlur={() => setEditing(null)}
                />
              ) : (
                <Text variant="body-medium">{row.value}</Text>
              )}
              {!isLive && editing !== index && (
                <Button
                  variant="ghost"
                  size="small"
                  aria-label={`Edit ${row.label.toLowerCase()}`}
                  disabled={isLeaving}
                  onClick={() => setEditing(index)}
                >
                  Edit
                </Button>
              )}
            </dd>
          </div>
        ))}
      </dl>
      {!isLive && onConfirm && (
        <Button
          variant="primary"
          size="large"
          loading={isLeaving}
          disabled={isLeaving}
          onClick={() => onConfirm(profileCorrections(rows, edited))}
        >
          That&apos;s me
        </Button>
      )}
    </section>
  );
}
