import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { okData } from "@/app/api/helpers";
import { useState } from "react";
import { isOpenTask } from "../../../task-helpers";

interface Args {
  enabled: boolean;
}

export function useDelegatedTasksBoard({ enabled }: Args) {
  const [openTaskId, setOpenTaskId] = useState<string | null>(null);

  const tasksQuery = useListTasks(undefined, {
    query: { select: (res) => okData(res) ?? null, enabled },
  });

  const tasks: DelegatedTask[] = tasksQuery.data ?? [];

  return {
    activeTasks: tasks.filter(isOpenTask),
    historyTasks: tasks.filter((task) => !isOpenTask(task)),
    isLoading: enabled && tasksQuery.isLoading,
    isError: tasksQuery.isError && tasksQuery.data == null,
    refetch: () => tasksQuery.refetch(),
    openTaskId,
    openTask: (taskId: string) => setOpenTaskId(taskId),
    closeTask: () => setOpenTaskId(null),
  };
}
