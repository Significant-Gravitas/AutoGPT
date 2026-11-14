import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";

const WORKING_POLL_MS = 5_000;
const IDLE_POLL_MS = 20_000;
/** Otto's tasks have no owner. */
const OTTO = "otto";

/** How many tasks each expert is working on right now, so the sidebar can
 *  mark who is busy. Checked often while someone works, now and then while
 *  nobody does, so work started elsewhere still shows up. */
export function useWorkingExperts() {
  const tasksQuery = useListTasks(
    { status: TaskStatus.IN_PROGRESS, include_subtasks: true, limit: 100 },
    {
      query: {
        select: (res) => (res.status === 200 ? res.data.tasks : []),
        refetchInterval: (query) => {
          const res = query.state.data;
          const isWorking = res?.status === 200 && res.data.tasks.length > 0;
          return isWorking ? WORKING_POLL_MS : IDLE_POLL_MS;
        },
      },
    },
  );
  const counts = new Map<string, number>();
  for (const task of tasksQuery.data ?? []) {
    const owner = task.owner_expert_id ?? OTTO;
    counts.set(owner, (counts.get(owner) ?? 0) + 1);
  }

  function workingCount(expertID: string | null) {
    return counts.get(expertID ?? OTTO) ?? 0;
  }

  return { workingCount };
}
