import { useState } from "react";
import type { DeletionSource, ExitReason } from "../../helpers";
import {
  buildAnswerFeedback,
  getReasonFromLabel,
  getReasonLabel,
  getTextPlaceholder,
  SKIPPED_FEEDBACK,
} from "./helpers";
import { useExitFeedback } from "./useExitFeedback";
import { useReasonPanel } from "./useReasonPanel";

interface Args {
  origin: DeletionSource;
  onClose: () => void;
  onContinue: () => void;
}

export function useExitReasonStep({ origin, onClose, onContinue }: Args) {
  const [reason, setReason] = useState<ExitReason | null>(null);
  const [text, setText] = useState("");
  const { sendAnswer } = useExitFeedback();
  const { panel, isActionPending, runAction } = useReasonPanel({
    reason,
    origin,
    onClose,
  });

  function selectReason(label: string) {
    setReason(getReasonFromLabel(label));
  }

  function skip() {
    sendAnswer(SKIPPED_FEEDBACK);
    onContinue();
  }

  function submit() {
    sendAnswer(buildAnswerFeedback({ reason, text }));
    onContinue();
  }

  return {
    reasonLabel: getReasonLabel(reason),
    placeholder: getTextPlaceholder(reason),
    text,
    setText,
    panel,
    isActionPending,
    selectReason,
    runAction,
    skip,
    submit,
  };
}
