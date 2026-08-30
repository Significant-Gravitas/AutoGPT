import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { okData } from "@/app/api/helpers";
import { isOpenTask } from "../../../task-helpers";

interface Args {
  expertId: string;
  enabled: boolean;
}

export function useExpertTasksSection({ expertId, enabled }: Args) {
  const tasksQuery = useListTasks(
    { expert_id: expertId },
    { query: { select: (res) => okData(res) ?? null, enabled } },
  );

  const tasks: DelegatedTask[] = tasksQuery.data ?? [];

  return {
    activeTasks: tasks.filter(isOpenTask),
    historyTasks: tasks.filter((task) => !isOpenTask(task)),
    isLoading: enabled && tasksQuery.isLoading,
    // A stale list still beats an error card: only a fetch that produced
    // nothing at all is worth replacing the whole section.
    isError: tasksQuery.isError && tasksQuery.data == null,
    refetch: () => tasksQuery.refetch(),
  };
}
