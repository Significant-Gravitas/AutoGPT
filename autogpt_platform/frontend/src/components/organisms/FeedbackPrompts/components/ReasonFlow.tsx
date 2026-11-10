"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
import { useState } from "react";
import { MAX_FEEDBACK_NOTE_CHARS, OTHER_VALUE } from "../helpers";
import { ReasonTiles, type Reason } from "./ReasonTiles";
import { ThanksLine } from "./ThanksLine";

interface Props {
  reasons: Reason[];
  label: string;
  notePlaceholder?: string;
  size?: "default" | "compact";
  thanks?: string;
  initialReason?: string | null;
  initialSent?: boolean;
  onReason?: (value: string) => void;
  onSubmit?: (answer: { reason: string; note: string }) => void;
}

export function ReasonFlow({
  reasons,
  label,
  notePlaceholder = "Tell us a bit more (optional)",
  size = "default",
  thanks,
  initialReason = null,
  initialSent = false,
  onReason,
  onSubmit,
}: Props) {
  const [reason, setReason] = useState<string | null>(initialReason);
  const [note, setNote] = useState("");
  const [sent, setSent] = useState(initialSent);

  function pickReason(value: string) {
    setReason(value);
    onReason?.(value);
  }

  function send() {
    setSent(true);
    if (reason)
      onSubmit?.({
        reason,
        note: reason === OTHER_VALUE ? note.trim() : "",
      });
  }

  if (sent) return <ThanksLine>{thanks}</ThanksLine>;

  return (
    <div className="flex flex-col gap-3">
      <ReasonTiles
        reasons={reasons}
        label={label}
        selected={reason}
        onSelect={pickReason}
        size={size}
      />
      {reason === OTHER_VALUE ? (
        <Textarea
          maxLength={MAX_FEEDBACK_NOTE_CHARS}
          autoFocus
          rows={2}
          value={note}
          onChange={(e) => setNote(e.target.value)}
          aria-label={notePlaceholder}
          placeholder={notePlaceholder}
          className="min-h-0 resize-none"
        />
      ) : null}
      {reason ? (
        <div className="flex justify-end">
          <Button size="small" onClick={send}>
            Send
          </Button>
        </div>
      ) : null}
    </div>
  );
}
