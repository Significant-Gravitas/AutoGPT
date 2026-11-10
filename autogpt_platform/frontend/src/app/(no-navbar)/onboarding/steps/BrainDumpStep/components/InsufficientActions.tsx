"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Mic01Icon, PencilEdit01Icon } from "@hugeicons/core-free-icons";
import { actionLabels, type InsufficientMode } from "./insufficient-copy";

interface Props {
  mode: InsufficientMode;
  canRecord: boolean;
  onTalk: () => void;
  onWrite: () => void;
  onSkip: () => void;
}

const ACTION_CLASS = "flex-1 md:min-w-40 md:flex-none";

// Talk leads after a voice take and Write after a typed one; the primary
// one comes first. With the microphone blocked there is no Talk at all.
export function InsufficientActions({
  mode,
  canRecord,
  onTalk,
  onWrite,
  onSkip,
}: Props) {
  const labels = actionLabels(mode);
  const talkLeads = mode === "voice" && canRecord;

  const talk = canRecord && (
    <Button
      key="talk"
      type="button"
      variant={talkLeads ? "primary" : "secondary"}
      size="large"
      leadingIcon={Mic01Icon}
      onClick={onTalk}
      className={ACTION_CLASS}
    >
      {labels.talk}
    </Button>
  );
  const write = (
    <Button
      key="write"
      type="button"
      variant={talkLeads ? "secondary" : "primary"}
      size="large"
      leadingIcon={PencilEdit01Icon}
      onClick={onWrite}
      className={ACTION_CLASS}
    >
      {labels.write}
    </Button>
  );

  return (
    <div className="flex w-full flex-col items-center gap-1.5">
      <div className="flex w-full justify-center gap-3">
        {talkLeads ? [talk, write] : [write, talk]}
      </div>
      <Button
        type="button"
        variant="ghost"
        tone="secondary"
        size="small"
        onClick={onSkip}
      >
        Continue without it
      </Button>
    </div>
  );
}
