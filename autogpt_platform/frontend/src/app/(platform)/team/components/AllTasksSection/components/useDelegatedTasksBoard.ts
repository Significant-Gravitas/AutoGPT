import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { okData } from "@/app/api/helpers";

interface Args {
  enabled: boolean;
  /** Scope the board to one expert's tasks (the expert detail page). */
  expertId?: string;
}

export function useDelegatedTasksBoard({ enabled, expertId }: Args) {
  const tasksQuery = useListTasks(
    expertId ? { expert_id: expertId } : undefined,
    {
      query: { select: (res) => okData(res) ?? null, enabled },
    },
  );

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
