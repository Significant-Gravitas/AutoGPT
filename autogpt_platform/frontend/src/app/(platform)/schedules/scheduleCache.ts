import type { QueryClient } from "@tanstack/react-query";

import {
  getGetV1ListExecutionSchedulesForAUserQueryKey,
  getListUpcomingScheduleRunsQueryKey,
  type getV1ListExecutionSchedulesForAUserResponse,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import {
  getListTasksQueryKey,
  type listTasksResponse,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { Task } from "@/app/api/__generated__/models/task";

/** The task list the Schedules page reads. */
export const TASK_LIST_PARAMS = { limit: 100, include_subtasks: false };

const tasksKey = () => getListTasksQueryKey(TASK_LIST_PARAMS);
const workflowsKey = () => getGetV1ListExecutionSchedulesForAUserQueryKey();

/** What the lists held before a change, to put back if it fails. */
export interface Snapshot {
  tasks: listTasksResponse | undefined;
  workflows: getV1ListExecutionSchedulesForAUserResponse | undefined;
}

/** Stops the lists' in-flight reads (they would overwrite the change) and
 *  keeps what they hold now. */
export async function snapshot(queryClient: QueryClient): Promise<Snapshot> {
  await Promise.all([
    queryClient.cancelQueries({ queryKey: tasksKey() }),
    queryClient.cancelQueries({ queryKey: workflowsKey() }),
  ]);
  return {
    tasks: queryClient.getQueryData<listTasksResponse>(tasksKey()),
    workflows:
      queryClient.getQueryData<getV1ListExecutionSchedulesForAUserResponse>(
        workflowsKey(),
      ),
  };
}

export function restore(queryClient: QueryClient, previous: Snapshot) {
  queryClient.setQueryData(tasksKey(), previous.tasks);
  queryClient.setQueryData(workflowsKey(), previous.workflows);
}

/** Changes the cached task list at once, before the server answers. */
export function patchTasks(
  queryClient: QueryClient,
  change: (tasks: Task[]) => Task[],
) {
  queryClient.setQueryData<listTasksResponse>(tasksKey(), (old) =>
    old?.status === 200
      ? { ...old, data: { ...old.data, tasks: change(old.data.tasks) } }
      : old,
  );
}

/** Changes the cached workflow schedules at once. */
export function patchWorkflows(
  queryClient: QueryClient,
  change: (schedules: GraphExecutionJobInfo[]) => GraphExecutionJobInfo[],
) {
  queryClient.setQueryData<getV1ListExecutionSchedulesForAUserResponse>(
    workflowsKey(),
    (old) => (old?.status === 200 ? { ...old, data: change(old.data) } : old),
  );
}

/** Reads every list again, whatever the change did. */
export function refetchSchedules(queryClient: QueryClient) {
  queryClient.invalidateQueries({ queryKey: tasksKey() });
  queryClient.invalidateQueries({ queryKey: workflowsKey() });
  queryClient.invalidateQueries({
    queryKey: getListUpcomingScheduleRunsQueryKey(),
  });
}

export function updateTask(
  id: string,
  change: (task: Task) => Task,
): (tasks: Task[]) => Task[] {
  return (tasks) => tasks.map((task) => (task.id === id ? change(task) : task));
}
