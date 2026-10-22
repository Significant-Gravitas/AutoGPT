import {
  getGetBrainDumpSuggestedTasksQueryKey,
  type getBrainDumpSuggestedTasksResponse,
  useGetBrainDumpSuggestedTasks,
  useUpdateBrainDumpSuggestedTasks,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import { trackFunnel } from "@/services/experts/experts-analytics";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast as sonnerToast } from "sonner";
import { toastSendFailure } from "../../helpers/sendFailureToast";
import {
  countOpenTasks,
  getShownTasks,
  getTaskState,
  mergeUpdatedTasks,
  withState,
} from "./helpers";

export type SuggestionSurface = "intro" | "home";

interface Args {
  surface: SuggestionSurface;
  /** Resolves to the chat the message went to, once it exists. */
  onSend: (message: string) => void | Promise<string | undefined | void>;
  /** False while the greeting is being written: the personalised list lands
   *  with it, and a read before then caches the generic set. */
  enabled: boolean;
}

const UNDO_TOAST_MS = 6000;
const QUERY_KEY = getGetBrainDumpSuggestedTasksQueryKey();

export function useSuggestedTasks({ surface, onSend, enabled }: Args) {
  const queryClient = useQueryClient();
  const [sendingId, setSendingId] = useState<string | null>(null);

  const { data: response } = useGetBrainDumpSuggestedTasks({
    query: { enabled },
  });
  const list = response?.status === 200 ? response.data : null;
  const tasks = list?.tasks ?? [];
  const shownTasks = getShownTasks(tasks);

  // One scope, so writes reach the server in the order they were made even
  // when the block that made them has already collapsed.
  const { mutate } = useUpdateBrainDumpSuggestedTasks({
    mutation: {
      scope: { id: "brain-dump-suggested-tasks" },
      // Each write sets a state, so a retry is safe; giving up on a sent row
      // would reopen it and let the next click start a second chat.
      retry: 3,
      onSuccess: (result, { data }) => {
        if (result.status !== 200) return;
        updateCache(
          (current) =>
            mergeUpdatedTasks({
              tasks: current,
              updated: result.data.tasks ?? [],
              ids: data.task_ids,
            }),
          result.data.stored,
        );
      },
      onError: () => {
        void queryClient.invalidateQueries({ queryKey: QUERY_KEY });
        toast({
          title: "Couldn't update your suggestions",
          description: "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  function updateCache(
    next: (tasks: SuggestedTask[]) => SuggestedTask[],
    stored?: boolean,
  ) {
    queryClient.setQueryData<getBrainDumpSuggestedTasksResponse>(
      QUERY_KEY,
      (current) => {
        if (!current || current.status !== 200) return current;
        return {
          ...current,
          data: {
            tasks: next(current.data.tasks ?? []),
            stored: stored ?? current.data.stored,
          },
        };
      },
    );
  }

  async function sendTask(task: SuggestedTask) {
    if (sendingId) return;
    setSendingId(task.id);
    trackFunnel("home_suggestion_clicked", {
      position: shownTasks.indexOf(task),
      surface,
      remaining: countOpenTasks(tasks) - 1,
    });
    try {
      const sessionId = await onSend(task.prompt);
      // No chat came of it (a locked composer, say): the row stays open.
      if (!sessionId) return;
      updateCache((current) =>
        withState({ tasks: current, ids: [task.id], state: "sent" }),
      );
      mutate({
        data: { task_ids: [task.id], state: "sent", session_id: sessionId },
      });
    } catch (error) {
      toastSendFailure({
        title: "Could not start that task",
        error,
        recovery: "it is still in your suggestions",
      });
    } finally {
      setSendingId(null);
    }
  }

  function hideTasks(hidden: SuggestedTask[], scope: "one" | "all") {
    const ids = hidden.map((task) => task.id);
    const before = tasks;
    const hiddenOpen = hidden.filter(
      (task) => getTaskState(task) === "open",
    ).length;
    trackFunnel("home_suggestion_dismissed", {
      position: scope === "one" ? shownTasks.indexOf(hidden[0]) : null,
      surface,
      remaining: countOpenTasks(tasks) - hiddenOpen,
      scope,
    });
    updateCache((current) =>
      withState({ tasks: current, ids, state: "dismissed" }),
    );
    mutate({ data: { task_ids: ids, state: "dismissed" } });

    const toastId = `suggestion-hidden-${ids.join("-")}`;
    toast({
      title: scope === "all" ? "Suggestions hidden" : "Suggestion hidden",
      duration: UNDO_TOAST_MS,
      replaceKey: toastId,
      action: (
        <Button
          variant="secondary"
          size="xs"
          onClick={() => {
            sonnerToast.dismiss(toastId);
            undoHide(before, ids);
          }}
        >
          Undo
        </Button>
      ),
    });
  }

  // An Undo sends no event: the dismissal stands as a fact.
  function undoHide(before: SuggestedTask[], ids: string[]) {
    const previous = new Map(before.map((task) => [task.id, task]));
    updateCache((current) =>
      current.map((task) =>
        ids.includes(task.id) ? (previous.get(task.id) ?? task) : task,
      ),
    );
    mutate({ data: { task_ids: ids, state: "open" } });
  }

  return {
    isStored: Boolean(list?.stored),
    tasks: shownTasks,
    sendingId,
    sendTask,
    hideTask: (task: SuggestedTask) => hideTasks([task], "one"),
    hideAll: () => hideTasks(shownTasks, "all"),
  };
}
