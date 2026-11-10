import { usePostV2SubmitMessageFeedback } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  getGetV2GetFeedbackPromptQueryKey,
  type getV2GetFeedbackPromptResponse,
  useGetV2GetFeedbackPrompt,
} from "@/app/api/__generated__/endpoints/feedback-prompts/feedback-prompts";
import type { FeedbackPromptState } from "@/app/api/__generated__/models/feedbackPromptState";
import { useFeedbackPrompt } from "@/components/organisms/FeedbackPrompts/useFeedbackPrompt";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useQueryClient } from "@tanstack/react-query";
import { useRef } from "react";
import type { FeedbackCardTarget } from "./helpers";

interface Args {
  kind: "work_check" | "otto_mistake";
  promptId: string;
  target: FeedbackCardTarget;
}

// A chat card's state lives on its row, so a reload knows it was answered or
// closed. The rating itself goes through the reply's feedback path.
export function useFeedbackCard({ kind, promptId, target }: Args) {
  const queryClient = useQueryClient();
  const prompt = useFeedbackPrompt(kind);
  const scoreRef = useRef<0 | 1 | 2>(0);
  const { mutate: submitRating } = usePostV2SubmitMessageFeedback({
    mutation: {
      scope: { id: `feedback-card:${promptId}` },
      onSuccess(_response, { data }) {
        prompt.save({
          score: data.score_value,
          reason: data.reason,
          note: data.comment,
        });
        updateCard({
          answered: true,
          score: data.score_value,
          ...(data.reason ? { reason: data.reason } : {}),
        });
      },
    },
  });
  const { data: state, isPending } = useGetV2GetFeedbackPrompt(promptId, {
    query: {
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      select: (response) =>
        response.status === 200 ? response.data : undefined,
    },
  });

  useMountEffect(() => {
    prompt.adopt(promptId);
  });

  function updateCard(change: Partial<FeedbackPromptState>) {
    queryClient.setQueryData<getV2GetFeedbackPromptResponse>(
      getGetV2GetFeedbackPromptQueryKey(promptId),
      (current) =>
        current?.status === 200
          ? { ...current, data: { ...current.data, ...change } }
          : current,
    );
  }

  function sendRating(rating: { reason?: string; comment?: string }) {
    submitRating({
      sessionId: target.sessionID,
      data: {
        message_id: target.messageID,
        score_name: kind,
        score_value: scoreRef.current,
        ...rating,
      },
    });
  }

  function rate(score: 0 | 1 | 2) {
    scoreRef.current = score;
    sendRating({});
  }

  function pickReason(reason: string) {
    sendRating({ reason });
  }

  function submit({ reason, note }: { reason: string; note: string }) {
    sendRating({ reason, comment: note || undefined });
  }

  function dismiss() {
    prompt.save({ dismissed: true });
    updateCard({ dismissed: true });
  }

  return { state, isLoading: isPending, rate, pickReason, submit, dismiss };
}
