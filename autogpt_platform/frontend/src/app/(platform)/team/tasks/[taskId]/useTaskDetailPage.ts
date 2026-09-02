import {
  getListTasksQueryKey,
  useCancelTask,
  useGetTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";

interface Args {
  taskId: string;
  enabled: boolean;
}

export function useTaskDetailPage({ taskId, enabled }: Args) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const detailQuery = useGetTask(taskId, {
    query: { select: (res) => okData(res) ?? null, enabled },
  });

  const { mutate: cancelTask, isPending: isCancelling } = useCancelTask({
    mutation: {
      onSuccess: () => {
        // Cancelling cascades to descendants, so the list is refetched rather
        // than patched — several rows can change from one click.
        queryClient.invalidateQueries({ queryKey: getListTasksQueryKey() });
        detailQuery.refetch();
      },
      onError: () => {
        toast({ title: "Could not cancel this task", variant: "destructive" });
      },
    },
  });

  return {
    task: detailQuery.data?.task ?? null,
    children: detailQuery.data?.children ?? [],
    isLoading: enabled && detailQuery.isLoading,
    isError:
      detailQuery.isError || (detailQuery.isFetched && !detailQuery.data?.task),
    refetch: () => detailQuery.refetch(),
    cancel: () => cancelTask({ taskId }),
    isCancelling,
  };
}
