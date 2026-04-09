import { toast } from "@/components/molecules/Toast/use-toast";
import { environment } from "@/services/environment";
import { getCopilotAuthHeaders } from "@/app/(platform)/copilot/helpers";
import { useState } from "react";

interface Args {
  sessionID: string | null;
  messageID: string;
}

async function submitFeedbackToBackend(args: {
  sessionID: string;
  messageID: string;
  scoreName: string;
  scoreValue: number;
  comment?: string;
}) {
  try {
    const authHeaders = await getCopilotAuthHeaders();
    await fetch(
      `${environment.getAGPTServerBaseUrl()}/api/chat/sessions/${args.sessionID}/feedback`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders,
        },
        body: JSON.stringify({
          message_id: args.messageID,
          score_name: args.scoreName,
          score_value: args.scoreValue,
          comment: args.comment,
        }),
      },
    );
  } catch (err) {
    // Feedback submission is best-effort; silently ignore failures
    console.debug("[Copilot] Feedback submission failed:", err);
  }
}

export function useMessageFeedback({ sessionID, messageID }: Args) {
  const [feedback, setFeedback] = useState<"upvote" | "downvote" | null>(null);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);

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
    if (sessionID) {
      submitFeedbackToBackend({
        sessionID,
        messageID,
        scoreName: "copy",
        scoreValue: 1,
      });
    }
  }

  function handleUpvote() {
    if (feedback) return;
    setFeedback("upvote");
    toast({
      title: "Thank you for your feedback!",
      variant: "success",
      duration: 3000,
    });
    if (sessionID) {
      submitFeedbackToBackend({
        sessionID,
        messageID,
        scoreName: "user-feedback",
        scoreValue: 1,
      });
    }
  }

  function handleDownvoteClick() {
    if (feedback) return;
    setFeedback("downvote");
    setShowFeedbackModal(true);
  }

  function handleDownvoteSubmit(comment: string) {
    setShowFeedbackModal(false);
    if (sessionID) {
      submitFeedbackToBackend({
        sessionID,
        messageID,
        scoreName: "user-feedback",
        scoreValue: 0,
        comment: comment || undefined,
      });
    }
  }

  function handleDownvoteCancel() {
    setShowFeedbackModal(false);
    setFeedback(null);
  }

  return {
    feedback,
    showFeedbackModal,
    handleCopy,
    handleUpvote,
    handleDownvoteClick,
    handleDownvoteSubmit,
    handleDownvoteCancel,
  };
}
