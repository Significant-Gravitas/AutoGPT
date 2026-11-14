import type { CancelTrialRequest } from "@/app/api/__generated__/models/cancelTrialRequest";
import { CancelTrialRequestReason } from "@/app/api/__generated__/models/cancelTrialRequestReason";
import { useFeedbackPrompt } from "@/components/organisms/FeedbackPrompts/useFeedbackPrompt";
import type { CancelTrialAnswer } from "@/components/organisms/FeedbackPrompts/useCancelTrialFeedback";

const REASONS: readonly string[] = Object.values(CancelTrialRequestReason);

function toCancelReason(reason: string | null): CancelTrialRequestReason {
  return reason !== null && REASONS.includes(reason)
    ? (reason as CancelTrialRequestReason)
    : null;
}

// Where the trial cancel answers go: each tap to our feedback row, and the
// final answer to Stripe with the cancellation itself.
export function useCancelTrialPrompt() {
  const prompt = useFeedbackPrompt("trial_cancel");

  function handleOpen() {
    void prompt.claim(true);
  }

  function handleReason(reason: string) {
    prompt.save({ reason });
  }

  function toCancelRequest({ reason, note }: CancelTrialAnswer) {
    if (note) prompt.save({ note });
    const request: CancelTrialRequest = {
      reason: toCancelReason(reason),
      comment: note || null,
    };
    return request;
  }

  return { handleOpen, handleReason, toCancelRequest };
}
