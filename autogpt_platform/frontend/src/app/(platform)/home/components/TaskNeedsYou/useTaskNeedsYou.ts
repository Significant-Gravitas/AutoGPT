import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { okData } from "@/app/api/helpers";
import { ownerOf } from "@/app/(platform)/tasks/helpers";

/** Tasks waiting on the user's input, newest first, each with who has it. */
export function useTaskNeedsYou() {
  const tasksQuery = useListTasks(
    { status: TaskStatus.NEEDS_INPUT },
    { query: { select: (res) => okData(res)?.tasks ?? [] } },
  );
  const tasks = tasksQuery.data ?? [];
  const expertsQuery = useListExperts({
    query: {
      select: (res) => okData(res) ?? [],
      enabled: tasks.some((task) => task.owner_expert_id),
    },
  });
  const experts = expertsQuery.data ?? [];

  return {
    rows: tasks.map((task) => ({ task, owner: ownerOf(task, experts) })),
  };
}
