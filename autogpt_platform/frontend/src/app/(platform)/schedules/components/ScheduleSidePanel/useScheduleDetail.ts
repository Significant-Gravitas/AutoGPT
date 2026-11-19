"use client";

import { useState } from "react";

import { useGetV1ListGraphExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetTaskRecord } from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { Task } from "@/app/api/__generated__/models/task";
import { okData } from "@/app/api/helpers";

import { runHref } from "@/app/(platform)/runs/helpers";
import { taskHref } from "@/app/(platform)/tasks/helpers";

import type { PanelAction, Schedule } from "../../affect/agent-schedules";
import { executionHistory, parseRowID, taskRunHistory } from "../../helpers";

interface Args {
  row: string | null;
  schedules: Schedule[];
  tasks: Task[];
  workflows: GraphExecutionJobInfo[];
  onDelete: (row: string) => void;
  onOpen: (href: string) => void;
}

/** The open schedule with its last runs (the task's runs, or the workflow's
 *  last executions) and the actions that leave the page. */
export function useScheduleDetail(args: Args) {
  const [confirming, setConfirming] = useState<string | null>(null);
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

  const base = args.schedules.find((s) => s.id === args.row) ?? null;
  const history = task
    ? taskRunHistory(record.data?.runs ?? [])
    : executionHistory(executions.data?.executions ?? []);
  const schedule = base
    ? { ...base, history: history?.length ? history : base.history }
    : null;

  return {
    schedule,
    actions: [
      ...lastRunAction(args, workflow, executions.data?.executions ?? []),
      ...actionsFor(args, task, workflow, confirming, setConfirming),
    ],
  };
}

/** A workflow schedule's newest run, on the run's own page. */
function lastRunAction(
  args: Args,
  workflow: GraphExecutionJobInfo | undefined,
  executions: GraphExecutionMeta[],
): PanelAction[] {
  const last = executions[0];
  if (!workflow || !last) return [];
  return [
    {
      label: "Open last run",
      onClick: () => args.onOpen(runHref(workflow.graph_id, last.id)),
    },
  ];
}

function actionsFor(
  args: Args,
  task: Task | undefined,
  workflow: GraphExecutionJobInfo | undefined,
  confirming: string | null,
  setConfirming: (row: string | null) => void,
): PanelAction[] {
  const row = args.row;
  if (!row) return [];
  const href = task ? taskHref(task) : null;
  if (href)
    return [{ label: "Open in chat", onClick: () => args.onOpen(href) }];
  if (!workflow) return [];
  if (confirming !== row) {
    return [
      { label: "Delete", danger: true, onClick: () => setConfirming(row) },
    ];
  }
  return [
    {
      label: "Delete this schedule",
      danger: true,
      onClick: () => {
        setConfirming(null);
        args.onDelete(row);
      },
    },
    { label: "Keep", onClick: () => setConfirming(null) },
  ];
}
