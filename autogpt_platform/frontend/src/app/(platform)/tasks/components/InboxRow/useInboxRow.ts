import { toast } from "@/components/molecules/Toast/use-toast";
import { isComposingEvent } from "@/lib/keyboard";
import { useEffect, useState, type KeyboardEvent } from "react";
import type { Task } from "../../data/types";
import {
  useAnswerTaskQuestion,
  useStartTask,
  useUpdateTask,
} from "../../data/useTasksData";
import { elapsedFrom, isTypingTarget, rowAction } from "./helpers";

function showError(title: string) {
  return function onError(error: unknown) {
    toast({
      title,
      description: error instanceof Error ? error.message : undefined,
      variant: "destructive",
    });
  };
}

function useNow(ticking: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!ticking) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [ticking]);
  return now;
}

export function useInboxRow(task: Task, expanded: boolean) {
  const running = task.inbox.state === "running";
  const now = useNow(running);
  const answer = useAnswerTaskQuestion();
  const update = useUpdateTask();
  const start = useStartTask();
  const action = rowAction(task);

  const liveElapsed = running ? elapsedFrom(task.started_at, now) : null;
  const elapsed = liveElapsed ?? task.elapsed_ms;
  const busy = answer.isPending || update.isPending || start.isPending;

  function answerWith(option: string) {
    answer.mutate(
      { id: task.id, answer: option, inputId: task.open_question?.id },
      { onError: showError("Couldn't send your answer") },
    );
  }

  function approve() {
    update.mutate(
      { id: task.id, input: { status: "done" } },
      { onError: showError("Couldn't approve the task") },
    );
  }

  function reject() {
    update.mutate(
      { id: task.id, input: { status: "in_progress" } },
      { onError: showError("Couldn't send the task back") },
    );
  }

  function retry() {
    start.mutate(task.id, { onError: showError("Couldn't restart the task") });
  }

  function handleKeyDown(e: KeyboardEvent<HTMLElement>) {
    if (!expanded || action !== "review" || busy) return;
    if (isComposingEvent(e) || isTypingTarget(e.target)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key.toLowerCase() !== "a") return;
    e.preventDefault();
    approve();
  }

  return {
    now,
    elapsed,
    action,
    busy,
    answerWith,
    approve,
    reject,
    retry,
    handleKeyDown,
    pending: {
      answer: answer.isPending ? answer.variables?.answer : undefined,
      approve: update.isPending && update.variables?.input.status === "done",
      reject:
        update.isPending && update.variables?.input.status === "in_progress",
      retry: start.isPending,
    },
  };
}
