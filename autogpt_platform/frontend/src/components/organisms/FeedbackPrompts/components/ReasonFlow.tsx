"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
import { useState } from "react";
import { OTHER_LABEL } from "../helpers";
import { ReasonTiles, type Reason } from "./ReasonTiles";
import { ThanksLine } from "./ThanksLine";

interface Props {
  reasons: Reason[];
  notePlaceholder?: string;
  size?: "default" | "compact";
  thanks?: string;
  initialReason?: string | null;
  initialSent?: boolean;
}

export function ReasonFlow({
  reasons,
  notePlaceholder = "Tell us a bit more (optional)",
  size = "default",
  thanks,
  initialReason = null,
  initialSent = false,
}: Props) {
  const [reason, setReason] = useState<string | null>(initialReason);
  const [note, setNote] = useState("");
  const [sent, setSent] = useState(initialSent);

  if (sent) return <ThanksLine>{thanks}</ThanksLine>;

  return (
    <div className="flex flex-col gap-3">
      <ReasonTiles
        reasons={reasons}
        selected={reason}
        onSelect={setReason}
        size={size}
      />
      {reason === OTHER_LABEL ? (
        <Textarea
          autoFocus
          rows={2}
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder={notePlaceholder}
          className="min-h-0 resize-none"
        />
      ) : null}
      {reason ? (
        <div className="flex justify-end">
          <Button size="small" onClick={() => setSent(true)}>
            Send
          </Button>
        </div>
      ) : null}
    </div>
  );
}
