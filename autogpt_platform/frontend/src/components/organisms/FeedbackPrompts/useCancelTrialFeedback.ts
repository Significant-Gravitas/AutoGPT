import { useState } from "react";
import { OTHER_VALUE } from "./helpers";

export type Step = "reason" | "offer" | "confirm";

const OFFER_REASON = "no_results_yet";
const NOTE_REASONS = [OTHER_VALUE, "missing_something"];

interface Args {
  startsOpen: boolean;
  initialStep: Step;
  initialReason: string | null;
  onOpen?: () => void;
  onReason?: (value: string) => void;
  onNote?: (note: string) => void;
  onConfirm?: () => void;
  onPickStarter?: (starter: string) => void;
  onTryOtto?: () => void;
}

export function useCancelTrialFeedback({
  startsOpen,
  initialStep,
  initialReason,
  onOpen,
  onReason,
  onNote,
  onConfirm,
  onPickStarter,
  onTryOtto,
}: Args) {
  const [isOpen, setIsOpen] = useState(startsOpen);
  const [step, setStep] = useState<Step>(initialStep);
  const [reason, setReason] = useState<string | null>(initialReason);
  const [note, setNote] = useState("");
  const asksForNote = reason !== null && NOTE_REASONS.includes(reason);

  function setOpen(open: boolean) {
    setIsOpen(open);
    if (!open) return;
    setStep(initialStep);
    setReason(initialReason);
    setNote("");
    onOpen?.();
  }

  function pickReason(value: string) {
    setReason(value);
    onReason?.(value);
  }

  function continueFromReason() {
    if (asksForNote && note.trim()) onNote?.(note.trim());
    setStep(reason === OFFER_REASON ? "offer" : "confirm");
  }

  function confirm() {
    setIsOpen(false);
    onConfirm?.();
  }

  function pickStarter(starter: string) {
    setIsOpen(false);
    onPickStarter?.(starter);
  }

  function tryOtto() {
    setIsOpen(false);
    onTryOtto?.();
  }

  return {
    isOpen,
    step,
    reason,
    note,
    asksForNote,
    setOpen,
    setStep,
    setNote,
    pickReason,
    continueFromReason,
    confirm,
    pickStarter,
    tryOtto,
  };
}
