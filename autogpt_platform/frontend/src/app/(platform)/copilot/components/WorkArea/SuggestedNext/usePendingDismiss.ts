import {
  dismissTaskSuggestion,
  getListTaskSuggestionsQueryKey,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

/** How long "Not now" can be taken back before it is sent. */
export const UNDO_MS = 5000;

/** "Not now" on a suggestion: it leaves at once, and is only sent once the
 *  undo has run out, before more are drafted, or when the brief closes. */
export function usePendingDismiss() {
  const queryClient = useQueryClient();
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const [hidden, setHidden] = useState<string[]>([]);
  const [last, setLast] = useState<Task | null>(null);

  function send(taskID: string) {
    timers.current.delete(taskID);
    return dismissTaskSuggestion(taskID).finally(() =>
      queryClient.invalidateQueries({
        queryKey: getListTaskSuggestionsQueryKey(),
      }),
    );
  }

  function setAside(task: Task) {
    setHidden((ids) => [...ids, task.id]);
    setLast(task);
    const timer = setTimeout(() => {
      void send(task.id);
      setLast((current) => (current?.id === task.id ? null : current));
    }, UNDO_MS);
    timers.current.set(task.id, timer);
  }

  function undo() {
    if (!last) return;
    clearTimeout(timers.current.get(last.id));
    timers.current.delete(last.id);
    setHidden((ids) => ids.filter((id) => id !== last.id));
    setLast(null);
  }

  /** Send every dismissal still waiting out its undo. */
  async function flush() {
    const waiting = [...timers.current.entries()];
    waiting.forEach(([, timer]) => clearTimeout(timer));
    setLast(null);
    await Promise.allSettled(waiting.map(([taskID]) => send(taskID)));
  }

  useEffect(() => {
    const pending = timers.current;
    return () => {
      pending.forEach((timer, taskID) => {
        clearTimeout(timer);
        void dismissTaskSuggestion(taskID).finally(() =>
          queryClient.invalidateQueries({
            queryKey: getListTaskSuggestionsQueryKey(),
          }),
        );
      });
      pending.clear();
    };
  }, [queryClient]);

  return { hidden, last, setAside, undo, flush };
}
