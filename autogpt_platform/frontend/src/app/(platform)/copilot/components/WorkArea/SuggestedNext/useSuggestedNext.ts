import {
  getListTaskSuggestionsQueryKey,
  getListTasksQueryKey,
  useRefreshTaskSuggestions,
  useRunTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useTaskSuggestions } from "../../EmptySession/components/HomeToday/useTaskSuggestions";
import { visibleCards } from "./helpers";
import { usePendingDismiss } from "./usePendingDismiss";

/** The most suggestions Otto keeps waiting at once. */
const MAX_SUGGESTIONS = 3;

function showError(title: string, error: unknown) {
  toast({
    title,
    description: error instanceof Error ? error.message : undefined,
    variant: "destructive",
  });
}

/** The brief's "Suggested next": Otto's drafts, started with one click (the
 *  card then stands for the running task), set aside with an undo, or topped
 *  up on request. */
export function useSuggestedNext() {
  const queryClient = useQueryClient();
  const { suggestions } = useTaskSuggestions();
  const dismissal = usePendingDismiss();
  const [started, setStarted] = useState<Task[]>([]);
  const [isUpToDate, setIsUpToDate] = useState(false);

  const run = useRunTask({
    mutation: {
      onSuccess: (res) => {
        const task = okData(res);
        if (task) setStarted((tasks) => [...tasks, task]);
        void queryClient.invalidateQueries({
          queryKey: getListTasksQueryKey(),
        });
      },
      onError: (error) => showError("Couldn't start the task", error),
    },
  });
  const refresh = useRefreshTaskSuggestions({
    mutation: {
      onSuccess: (res) => {
        const drafted = okData(res)?.tasks ?? [];
        setIsUpToDate(drafted.every((task) => isKnown(task.id)));
        void queryClient.invalidateQueries({
          queryKey: getListTaskSuggestionsQueryKey(),
        });
      },
      onError: (error) => showError("Couldn't suggest more tasks", error),
    },
  });

  function isKnown(taskID: string) {
    return suggestions.some((task) => task.id === taskID);
  }

  const cards = visibleCards(suggestions, started, dismissal.hidden);
  const waiting = cards.filter((task) => task.is_suggested).length;

  async function suggestMore() {
    setIsUpToDate(false);
    await dismissal.flush();
    refresh.mutate();
  }

  return {
    cards,
    lastSetAside: dismissal.last,
    canSuggestMore: waiting < MAX_SUGGESTIONS,
    isSuggesting: refresh.isPending,
    isUpToDate,
    startingID: run.isPending ? run.variables?.taskId : undefined,
    start: (task: Task) => run.mutate({ taskId: task.id, data: null }),
    setAside: dismissal.setAside,
    undo: dismissal.undo,
    suggestMore,
  };
}
