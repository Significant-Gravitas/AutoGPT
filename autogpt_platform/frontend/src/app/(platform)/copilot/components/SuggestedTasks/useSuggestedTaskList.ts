import {
  getGetBrainDumpSuggestedTasksQueryKey,
  type getBrainDumpSuggestedTasksResponse,
  useGetBrainDumpSuggestedTasks,
  useUpdateBrainDumpSuggestedTasks,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { mergeUpdatedTasks } from "./helpers";

const QUERY_KEY = getGetBrainDumpSuggestedTasksQueryKey();

// The stored list and its writes. Changes are applied to the cache first and
// written in the order they were made.
export function useSuggestedTaskList({ enabled }: { enabled: boolean }) {
  const queryClient = useQueryClient();
  const { data: response } = useGetBrainDumpSuggestedTasks({
    query: { enabled },
  });
  const list = response?.status === 200 ? response.data : null;
  const tasks = list?.tasks ?? [];

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

  return { list, tasks, mutate, updateCache };
}
