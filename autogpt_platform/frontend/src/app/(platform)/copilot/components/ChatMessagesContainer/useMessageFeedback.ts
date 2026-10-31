import { usePostV2SubmitMessageFeedback } from "@/app/api/__generated__/endpoints/chat/chat";
import type { MessageFeedbackRequest } from "@/app/api/__generated__/models/messageFeedbackRequest";
import { toast } from "@/components/molecules/Toast/use-toast";
import * as Sentry from "@sentry/nextjs";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useState } from "react";
import { extractDbSequence } from "../../helpers/convertChatSessionToUiMessages";

type Rating = "upvote" | "downvote";

interface Args {
  sessionID: string | null;
  message: UIMessage<unknown, UIDataTypes, UITools>;
  savedMessageId?: string | null;
}

export function useMessageFeedback({
  sessionID,
  message,
  savedMessageId,
}: Args) {
  const [feedback, setFeedback] = useState<Rating | null>(null);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const { mutateAsync: submitFeedback } = usePostV2SubmitMessageFeedback();

  // Only a saved reply can be rated. A reply that just streamed is keyed by
  // the stream until the chat swaps in its saved id or learns it.
  const ratedMessageId =
    savedMessageId !== undefined
      ? savedMessageId
      : extractDbSequence(message) !== null
        ? message.id
        : null;
  const canRate = sessionID !== null && ratedMessageId !== null;

  async function sendScore(score: Omit<MessageFeedbackRequest, "message_id">) {
    if (!sessionID || !ratedMessageId) return false;
    try {
      await submitFeedback({
        sessionId: sessionID,
        data: { message_id: ratedMessageId, ...score },
      });
      return true;
    } catch (error) {
      Sentry.captureException(error);
      return false;
    }
  }

  async function saveRating(rating: Rating, comment?: string) {
    const saved = await sendScore({
      score_name: "user-feedback",
      score_value: rating === "upvote" ? 1 : 0,
      comment,
    });
    if (!saved) {
      setFeedback(null);
      toast({
        title: "Couldn't save your feedback",
        description: "Please try again.",
        variant: "destructive",
      });
      return;
    }
    toast({
      title: "Thank you for your feedback!",
      variant: "success",
      duration: 3000,
    });
  }

  async function handleCopy(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      toast({ title: "Copied!", variant: "success", duration: 2000 });
    } catch {
      toast({
        title: "Failed to copy",
        variant: "destructive",
        duration: 2000,
      });
      return;
    }
    // A signal for us, not an action the user asked for: the copy itself
    // succeeded, so a failure to record it is reported but not shown.
    if (canRate) void sendScore({ score_name: "copy", score_value: 1 });
  }

  function explainNotRateableYet() {
    toast({
      title: "Can't rate this reply yet",
      description:
        "It may still be saving. Try again in a moment, or refresh the page.",
      duration: 3000,
    });
  }

  function handleUpvote() {
    if (feedback) return;
    if (!canRate) return explainNotRateableYet();
    setFeedback("upvote");
    void saveRating("upvote");
  }

  function handleDownvoteClick() {
    if (feedback) return;
    if (!canRate) return explainNotRateableYet();
    setFeedback("downvote");
    setShowFeedbackModal(true);
  }

  function handleDownvoteSubmit(comment: string) {
    setShowFeedbackModal(false);
    void saveRating("downvote", comment || undefined);
  }

  function handleDownvoteCancel() {
    setShowFeedbackModal(false);
    setFeedback(null);
  }

  return {
    feedback,
    canRate,
    showFeedbackModal,
    handleCopy,
    handleUpvote,
    handleDownvoteClick,
    handleDownvoteSubmit,
    handleDownvoteCancel,
  };
}
