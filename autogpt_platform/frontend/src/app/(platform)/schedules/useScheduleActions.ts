"use client";

import { useQueryClient } from "@tanstack/react-query";

import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV1ListExecutionSchedulesForAUserQueryKey,
  getListUpcomingScheduleRunsQueryKey,
  useDeleteV1DeleteExecutionSchedule,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import {
  getListTasksQueryKey,
  usePauseTaskRepeat,
  useResumeTaskRepeat,
  useRunTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { toast } from "@/components/molecules/Toast/use-toast";

import { parseRowID } from "./helpers";
import { TASK_LIST_PARAMS } from "./useSchedulesPage";

interface Args {
  workflows: GraphExecutionJobInfo[];
}

/** What a schedule can do from the page: pause or resume a repeating task,
 *  run a task or a workflow schedule now, and delete a workflow schedule. */
export function useScheduleActions({ workflows }: Args) {
  const queryClient = useQueryClient();

  function refresh() {
    queryClient.invalidateQueries({
      queryKey: getListTasksQueryKey(TASK_LIST_PARAMS),
    });
    queryClient.invalidateQueries({
      queryKey: getGetV1ListExecutionSchedulesForAUserQueryKey(),
    });
    queryClient.invalidateQueries({
      queryKey: getListUpcomingScheduleRunsQueryKey(),
    });
  }

  function failed(title: string) {
    return (error: unknown) =>
      toast({
        title,
        description: error instanceof Error ? error.message : undefined,
        variant: "destructive",
      });
  }

  const mutation = (title: string) => ({
    mutation: { onSuccess: refresh, onError: failed(title) },
  });
  const pause = usePauseTaskRepeat(mutation("The task could not be paused"));
  const resume = useResumeTaskRepeat(mutation("The task could not resume"));
  const runTask = useRunTask(mutation("The task could not start"));
  const runWorkflow = usePostV1ExecuteGraphAgent(
    mutation("The workflow could not start"),
  );
  const remove = useDeleteV1DeleteExecutionSchedule(
    mutation("The schedule could not be deleted"),
  );

  function setPaused(row: string, paused: boolean) {
    const target = parseRowID(row);
    if (target?.kind !== "task") return;
    (paused ? pause : resume).mutate({ taskId: target.id });
  }

  function runNow(row: string) {
    const target = parseRowID(row);
    if (!target) return;
    if (target.kind === "task") return runTask.mutate({ taskId: target.id });
    const schedule = workflows.find((s) => s.id === target.id);
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

  function deleteSchedule(row: string) {
    const target = parseRowID(row);
    if (target?.kind !== "workflow") return;
    remove.mutate({ scheduleId: target.id });
  }

  return { setPaused, runNow, deleteSchedule };
}
