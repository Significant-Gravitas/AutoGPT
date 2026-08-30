import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { okData } from "@/app/api/helpers";

interface Args {
  enabled: boolean;
}

export function useDelegatedTasksBoard({ enabled }: Args) {
  const tasksQuery = useListTasks(undefined, {
    query: { select: (res) => okData(res) ?? null, enabled },
  });

  const tasks: DelegatedTask[] = tasksQuery.data ?? [];

  return {
    tasks,
    isLoading: enabled && tasksQuery.isLoading,
    isError: tasksQuery.isError && tasksQuery.data == null,
    refetch: () => tasksQuery.refetch(),
    // The first load has its own skeleton, so the icon only spins for
    // refetches over rows that are already on screen.
    isRefreshing: tasksQuery.isFetching && !tasksQuery.isLoading,
  };
}
