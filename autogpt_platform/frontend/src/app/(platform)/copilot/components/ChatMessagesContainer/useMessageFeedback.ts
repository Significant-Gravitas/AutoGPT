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
}

export function useMessageFeedback({ sessionID, message }: Args) {
  const [feedback, setFeedback] = useState<Rating | null>(null);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const { mutateAsync: submitFeedback } = usePostV2SubmitMessageFeedback();

  // A reply that just streamed still carries the stream's id until the chat
  // reloads it with its saved one, and only the saved one can be rated.
  const canRate = sessionID !== null && extractDbSequence(message) !== null;

  async function sendScore(score: Omit<MessageFeedbackRequest, "message_id">) {
    if (!sessionID) return false;
    try {
      await submitFeedback({
        sessionId: sessionID,
        data: { message_id: message.id, ...score },
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

  function handleUpvote() {
    if (feedback || !canRate) return;
    setFeedback("upvote");
    void saveRating("upvote");
  }

  function handleDownvoteClick() {
    if (feedback || !canRate) return;
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
