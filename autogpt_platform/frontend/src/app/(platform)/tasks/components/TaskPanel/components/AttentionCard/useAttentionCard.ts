import { useRouter } from "next/navigation";
import { useState } from "react";
import { useAnswerAsk, useTaskAction } from "../../../../data/useTasksData";

export function useAttentionCard(taskId: string, goal: string) {
  const router = useRouter();
  const answerMutation = useAnswerAsk();
  const actionMutation = useTaskAction();
  const [answer, setAnswer] = useState("");
  const [newGoal, setNewGoal] = useState(goal);
  const [isEditingGoal, setIsEditingGoal] = useState(false);

  function sendAnswer(value: string, askId?: string) {
    const trimmed = value.trim();
    if (!trimmed || answerMutation.isPending) return;
    answerMutation.mutate(
      { id: taskId, answer: trimmed, askId },
      { onSuccess: () => setAnswer("") },
    );
  }

  function runAction(action: string) {
    if (action === "change_goal") {
      setIsEditingGoal(true);
      return;
    }
    actionMutation.mutate(
      { id: taskId, action },
      {
        onSuccess: (summary) => {
          if (action === "do_in_chat" && summary.work_session_id) {
            router.push(`/copilot?sessionId=${summary.work_session_id}`);
          }
        },
      },
    );
  }

  function submitGoal() {
    const trimmed = newGoal.trim();
    if (!trimmed || actionMutation.isPending) return;
    actionMutation.mutate(
      { id: taskId, action: "change_goal", body: { goal: trimmed } },
      { onSuccess: () => setIsEditingGoal(false) },
    );
  }

  return {
    answer,
    setAnswer,
    sendAnswer,
    pendingAnswer: answerMutation.isPending
      ? answerMutation.variables?.answer
      : undefined,
    isAnswering: answerMutation.isPending,
    runAction,
    pendingAction: actionMutation.isPending
      ? actionMutation.variables?.action
      : undefined,
    isActing: actionMutation.isPending,
    isEditingGoal,
    cancelGoal: () => setIsEditingGoal(false),
    newGoal,
    setNewGoal,
    submitGoal,
    error: answerMutation.error ?? actionMutation.error,
  };
}
