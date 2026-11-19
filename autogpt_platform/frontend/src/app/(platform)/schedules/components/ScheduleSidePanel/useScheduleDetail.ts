"use client";

import { useGetV1ListGraphExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetTaskRecord } from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { Task } from "@/app/api/__generated__/models/task";
import { okData } from "@/app/api/helpers";

import type { Schedule } from "../../affect/agent-schedules";
import { executionHistory, parseRowID, taskRunHistory } from "../../helpers";
import { armedLabels, panelActions } from "./scheduleActions";
import { useArmedConfirm } from "./useArmedConfirm";

interface Args {
  row: string | null;
  schedules: Schedule[];
  tasks: Task[];
  workflows: GraphExecutionJobInfo[];
  onRemove: (row: string) => void;
  onOpen: (href: string) => void;
}

/** The open schedule with its last runs (the task's runs, or the workflow's
 *  last executions) and the panel's own actions. */
export function useScheduleDetail(args: Args) {
  const target = args.row ? parseRowID(args.row) : null;
  const task =
    target?.kind === "task"
      ? args.tasks.find((t) => t.id === target.id)
      : undefined;
  const workflow =
    target?.kind === "workflow"
      ? args.workflows.find((w) => w.id === target.id)
      : undefined;

  const record = useGetTaskRecord(task?.id ?? "", {
    query: { enabled: Boolean(task), select: okData },
  });
  const executions = useGetV1ListGraphExecutions(
    workflow?.graph_id ?? "",
    { page: 1, page_size: 10 },
    { query: { enabled: Boolean(workflow), select: okData } },
  );

  const { confirming, setConfirming } = useArmedConfirm(
    args.row,
    armedLabels({ task, workflow }),
  );

  const base = args.schedules.find((s) => s.id === args.row) ?? null;
  const history = task
    ? taskRunHistory(record.data?.runs ?? [])
    : executionHistory(executions.data?.executions ?? []);
  const schedule = base
    ? { ...base, history: history?.length ? history : base.history }
    : null;

  return {
    schedule,
    actions: args.row
      ? panelActions({
          row: args.row,
          task,
          workflow,
          executions: executions.data?.executions ?? [],
          confirming,
          setConfirming,
          onRemove: args.onRemove,
          onOpen: args.onOpen,
        })
      : [],
  };
}
