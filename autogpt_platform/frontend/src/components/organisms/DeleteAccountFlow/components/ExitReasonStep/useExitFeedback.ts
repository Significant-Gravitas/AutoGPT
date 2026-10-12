import { usePostAccountSendExitFeedback } from "@/app/api/__generated__/endpoints/account/account";
import type { DeletionFeedbackRequest } from "@/app/api/__generated__/models/deletionFeedbackRequest";
import { useRef } from "react";
import type { ExitReason, PanelActionKind } from "../../helpers";

interface AlternativeArgs {
  reason: ExitReason;
  alternative: PanelActionKind;
}

// Feedback is fire and forget: nothing waits on it and a failed post never
// blocks leaving or deleting.
export function useExitFeedback() {
  const { mutate } = usePostAccountSendExitFeedback();
  const hasAnswered = useRef(false);

  function sendAnswer(data: DeletionFeedbackRequest) {
    if (hasAnswered.current) return;
    hasAnswered.current = true;
    mutate({ data });
  }

  function sendAlternative({ reason, alternative }: AlternativeArgs) {
    mutate({ data: { kind: "alternative_clicked", reason, alternative } });
  }

  return { sendAnswer, sendAlternative };
}
