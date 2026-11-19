"use client";

import { useQueryClient } from "@tanstack/react-query";

import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  useDeleteV1DeleteExecutionSchedule,
  usePostV1CreateExecutionSchedule,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import {
  useCancelTask,
  usePauseTaskRepeat,
  useResumeTaskRepeat,
  useRunTask,
  useStopTaskRepeat,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { Task } from "@/app/api/__generated__/models/task";

import { parseRowID } from "./helpers";
import {
  patchTasks,
  patchWorkflows,
  refetchSchedules,
  restore,
  snapshot,
  updateTask,
  type Snapshot,
} from "./scheduleCache";
import {
  failedToast,
  removedToast,
  startedToast,
  switchedToast,
} from "./scheduleToasts";

interface Args {
  tasks: Task[];
  workflows: GraphExecutionJobInfo[];
}

/** What a change kept to put the lists back, and what to say when it worked. */
interface Context {
  previous: Snapshot;
  done: string;
  /** A deleted workflow schedule, kept to put back with Undo. */
  deleted?: GraphExecutionJobInfo;
}

/** What a schedule can do from the page. Pause, resume, stop repeating,
 *  cancel and delete change the cached lists at once (onMutate), put them
 *  back with a toast if the server refuses (onError) and read the lists
 *  again either way (onSettled). Each says what it did; Run now says where
 *  the run went, and a deleted workflow schedule can be put back. */
export function useScheduleActions({ tasks, workflows }: Args) {
  const queryClient = useQueryClient();
  const taskOf = (id: string) => tasks.find((task) => task.id === id);
  const nameOf = (id: string) => taskOf(id)?.title ?? "The task";

  async function apply(
    done: string,
    change: () => void,
    deleted?: GraphExecutionJobInfo,
  ): Promise<Context> {
    const previous = await snapshot(queryClient);
    change();
    return { previous, done, deleted };
  }

  function optimistic(failure: string) {
    return {
      onError: (
        error: unknown,
        _vars: unknown,
        context: Context | undefined,
      ) => {
        if (context) restore(queryClient, context.previous);
        failedToast(failure, error);
      },
      onSettled: () => refetchSchedules(queryClient),
    };
  }

  const pause = usePauseTaskRepeat<unknown, Context>({
    mutation: {
      ...optimistic("The task could not be paused"),
      onMutate: ({ taskId }) =>
        apply(`Paused ${nameOf(taskId)}`, () =>
          patchTasks(queryClient, setRepeatPaused(taskId, true)),
        ),
      onSuccess: (_res, _vars, context) => switchedToast(context.done),
    },
  });
  const resume = useResumeTaskRepeat<unknown, Context>({
    mutation: {
      ...optimistic("The task could not resume"),
      onMutate: ({ taskId }) =>
        apply(`Resumed ${nameOf(taskId)}`, () =>
          patchTasks(queryClient, setRepeatPaused(taskId, false)),
        ),
      onSuccess: (_res, _vars, context) => switchedToast(context.done),
    },
  });
  const stopRepeat = useStopTaskRepeat<unknown, Context>({
    mutation: {
      ...optimistic("The task could not stop repeating"),
      onMutate: ({ taskId }) =>
        apply(
          taskOf(taskId)?.repeat?.watch
            ? `${nameOf(taskId)} stopped watching`
            : `${nameOf(taskId)} stopped repeating`,
          () =>
            patchTasks(
              queryClient,
              updateTask(taskId, (task) => ({
                ...task,
                repeat: null,
                start_at: null,
              })),
            ),
        ),
      onSuccess: (_res, _vars, context) => removedToast(context.done),
    },
  });
  const cancel = useCancelTask<unknown, Context>({
    mutation: {
      ...optimistic("The task could not be cancelled"),
      onMutate: ({ taskId }) =>
        apply(`${nameOf(taskId)} was cancelled`, () =>
          patchTasks(
            queryClient,
            updateTask(taskId, (task) => ({ ...task, status: "CANCELLED" })),
          ),
        ),
      onSuccess: (_res, _vars, context) => removedToast(context.done),
    },
  });
  const recreate = usePostV1CreateExecutionSchedule({
    mutation: {
      onSuccess: (_res, vars) => switchedToast(`${vars.data.name} is back`),
      onError: (error) =>
        failedToast("The schedule could not be restored", error),
      onSettled: () => refetchSchedules(queryClient),
    },
  });
  const remove = useDeleteV1DeleteExecutionSchedule<unknown, Context>({
    mutation: {
      ...optimistic("The schedule could not be deleted"),
      onMutate: ({ scheduleId }) => {
        const schedule = workflows.find((s) => s.id === scheduleId);
        return apply(
          `${schedule?.name ?? "The schedule"} was deleted`,
          () =>
            patchWorkflows(queryClient, (list) =>
              list.filter((s) => s.id !== scheduleId),
            ),
          schedule,
        );
      },
      // The schedule as it was before the change, from the context: by now
      // the list no longer holds it.
      onSuccess: (_res, _vars, { done, deleted }) =>
        removedToast(
          done,
          deleted ? () => recreate.mutate(recreation(deleted)) : undefined,
        ),
    },
  });
  const runTask = useRunTask({
    mutation: {
      onSuccess: (_res, { taskId }) =>
        startedToast(nameOf(taskId), { task: taskOf(taskId) }),
      onError: (error) => failedToast("The task could not start", error),
      onSettled: () => refetchSchedules(queryClient),
    },
  });
  const runWorkflow = usePostV1ExecuteGraphAgent({
    mutation: {
      onSuccess: (res, { graphId }) =>
        startedToast(
          workflows.find((s) => s.graph_id === graphId)?.name ?? "The workflow",
          {
            run:
              res.status === 200
                ? { graphID: graphId, runID: res.data.id }
                : undefined,
          },
        ),
      onError: (error) => failedToast("The workflow could not start", error),
      onSettled: () => refetchSchedules(queryClient),
    },
  });

  function setPaused(row: string, paused: boolean) {
    const target = parseRowID(row);
    if (target?.kind !== "task") return;
    (paused ? pause : resume).mutate({ taskId: target.id });
  }

  function runNow(row: string) {
    const target = parseRowID(row);
    if (target?.kind === "task") return runTask.mutate({ taskId: target.id });
    const schedule = workflows.find((s) => s.id === target?.id);
    if (!schedule) return;
    runWorkflow.mutate({
      graphId: schedule.graph_id,
      graphVersion: schedule.graph_version,
      data: {
        inputs: schedule.input_data,
        credentials_inputs: schedule.input_credentials ?? {},
        source: "library",
      },
    });
  }

  /** Delete a workflow schedule, stop a repeating or watching task, or
   *  cancel a task waiting for its one start. */
  function remove_(row: string) {
    const target = parseRowID(row);
    if (!target) return;
    if (target.kind === "workflow")
      return remove.mutate({ scheduleId: target.id });
    if (taskOf(target.id)?.repeat)
      return stopRepeat.mutate({ taskId: target.id });
    cancel.mutate({ taskId: target.id });
  }

  return { setPaused, runNow, remove: remove_ };
}

function setRepeatPaused(id: string, paused: boolean) {
  return updateTask(id, (task) =>
    task.repeat ? { ...task, repeat: { ...task.repeat, paused } } : task,
  );
}

/** The same schedule again: its workflow version, name, rhythm, timezone,
 *  inputs, accounts and expert. Only its id is new. */
function recreation(schedule: GraphExecutionJobInfo) {
  return {
    graphId: schedule.graph_id,
    data: {
      graph_version: schedule.graph_version,
      name: schedule.name,
      cron: schedule.cron,
      inputs: schedule.input_data,
      credentials: schedule.input_credentials ?? {},
      timezone: schedule.timezone,
      expert_id: schedule.expert_id ?? undefined,
    },
  };
}
