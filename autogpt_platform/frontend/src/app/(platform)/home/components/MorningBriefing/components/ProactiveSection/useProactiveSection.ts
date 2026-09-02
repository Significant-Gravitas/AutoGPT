import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { DelegatedTaskStatus } from "@/app/api/__generated__/models/delegatedTaskStatus";
import { okData } from "@/app/api/helpers";

const OPEN_STATUSES: DelegatedTaskStatus[] = [
  "QUEUED",
  "WORKING",
  "WAITING_USER",
];

export function useProactiveSection() {
  const tasksQuery = useListTasks(undefined, {
    query: { select: (res) => okData(res) ?? [] },
  });

  const dreamTasks: DelegatedTask[] = (tasksQuery.data ?? []).filter(
    (task) => task.created_by_type === "DREAM",
  );

  return {
    proposals: dreamTasks.filter((task) => OPEN_STATUSES.includes(task.status)),
    outcomes: dreamTasks.filter((task) => task.status === "DONE"),
  };
}
