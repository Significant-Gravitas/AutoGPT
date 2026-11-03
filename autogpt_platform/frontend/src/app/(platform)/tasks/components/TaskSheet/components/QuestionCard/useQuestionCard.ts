import { useState } from "react";
import { useAnswerTaskQuestion } from "../../../../data/useTasksData";

export function useQuestionCard(taskId: string, questionId: string) {
  const [answer, setAnswer] = useState("");
  const mutation = useAnswerTaskQuestion();

  function send(value: string) {
    const trimmed = value.trim();
    if (!trimmed || mutation.isPending) return;
    mutation.mutate(
      { id: taskId, answer: trimmed, inputId: questionId },
      { onSuccess: () => setAnswer("") },
    );
  }

  return {
    answer,
    setAnswer,
    send,
    isSending: mutation.isPending,
    pendingAnswer: mutation.isPending ? mutation.variables?.answer : undefined,
    isError: mutation.isError,
  };
}
