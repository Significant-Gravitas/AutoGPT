import { useState } from "react";
import { OTHER_VALUE } from "./helpers";

const NOTE_REASONS = [OTHER_VALUE, "missing_something"];

export interface CancelTrialAnswer {
  reason: string | null;
  note: string;
}

interface Args {
  startsOpen: boolean;
  initialReason: string | null;
  onOpen?: () => void;
  onReason?: (value: string) => void;
  onConfirm?: (answer: CancelTrialAnswer) => void;
}

export function useCancelTrialFeedback({
  startsOpen,
  initialReason,
  onOpen,
  onReason,
  onConfirm,
}: Args) {
  const [isOpen, setIsOpen] = useState(startsOpen);
  const [reason, setReason] = useState<string | null>(initialReason);
  const [note, setNote] = useState("");
  const asksForNote = reason !== null && NOTE_REASONS.includes(reason);

  function setOpen(open: boolean) {
    setIsOpen(open);
    if (!open) return;
    setReason(initialReason);
    setNote("");
    onOpen?.();
  }

  function pickReason(value: string) {
    setReason(value);
    onReason?.(value);
  }

  function confirm() {
    setIsOpen(false);
    onConfirm?.({ reason, note: asksForNote ? note.trim() : "" });
  }

  return {
    isOpen,
    reason,
    note,
    asksForNote,
    setOpen,
    setNote,
    pickReason,
    confirm,
  };
}
