import type { BriefingTaskItem } from "@/app/api/__generated__/models/briefingTaskItem";
import type { BriefingTasks } from "@/app/api/__generated__/models/briefingTasks";
import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";

export interface StoredBriefTasks {
  finished: BriefingTaskItem[];
  failed: BriefingTaskItem[];
  waiting: BriefingTaskItem[];
}

/** The tasks as this morning's brief wrote them down, or null when the brief
 *  predates tasks and the client has to join them in itself. A task it
 *  listed as waiting that has since been answered is left out. */
export function storedBriefTasks(
  stored: BriefingTasks | null | undefined,
  live: Task[],
): StoredBriefTasks | null {
  if (!stored) return null;
  const statusByID = new Map(live.map((task) => [task.id, task.status]));
  return {
    finished: stored.finished ?? [],
    failed: stored.failed ?? [],
    waiting: (stored.waiting ?? []).filter((item) => {
      const status = statusByID.get(item.task_id);
      return status === undefined || status === TaskStatus.NEEDS_INPUT;
    }),
  };
}
