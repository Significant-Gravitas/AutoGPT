"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { FilterTable } from "@/components/molecules/FilterTable/FilterTable";
import {
  TASK_TABLE_FILTERS,
  formatElapsed,
  formatSpend,
  getStatusLabel,
  getStatusVariant,
  getTaskFilterKey,
} from "../../../task-helpers";
import { TaskDetailDrawer } from "../../TaskDetailDrawer/TaskDetailDrawer";
import { useDelegatedTasksBoard } from "./useDelegatedTasksBoard";

interface Props {
  enabled: boolean;
}

const COLUMNS = [
  { key: "task", label: "Task name", width: "minmax(0,1.4fr)" },
  { key: "owner", label: "Owner", width: "minmax(0,0.9fr)" },
  { key: "status", label: "Status", width: "minmax(0,0.8fr)" },
  { key: "age", label: "Age", width: "minmax(0,0.5fr)" },
  { key: "spend", label: "Spend", width: "minmax(0,0.5fr)" },
];

/** The task-spine version of the team board: every DelegatedTask receipt
 *  across the whole team — including Autopilot work, which the run-based
 *  board can't see because it only fans out per hired expert. */
export function DelegatedTasksBoard({ enabled }: Props) {
  const {
    tasks,
    isLoading,
    isError,
    refetch,
    openTaskId,
    openTask,
    closeTask,
  } = useDelegatedTasksBoard({ enabled });

  if (isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-16 w-full rounded-2xl" />
        <Skeleton className="h-16 w-full rounded-2xl" />
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorCard
        context="your team's tasks"
        hint="We could not load what your team has been asked to do."
        onRetry={refetch}
      />
    );
  }

  if (tasks.length === 0) {
    return (
      <Text variant="body" className="text-zinc-500">
        Nothing delegated yet. Ask an expert (or Autopilot) to do something and
        it will show up here.
      </Text>
    );
  }

  return (
    <section aria-label="All tasks">
      <FilterTable
        ariaLabel="Delegated tasks"
        columns={COLUMNS}
        filters={TASK_TABLE_FILTERS}
        rows={tasks.map((task) => ({
          id: task.id,
          filterKey: getTaskFilterKey(task),
          onClick: () => openTask(task.id),
          cells: buildCells(task),
        }))}
      />

      <TaskDetailDrawer taskId={openTaskId} onClose={closeTask} />
    </section>
  );
}

function buildCells(task: DelegatedTask) {
  return {
    task: (
      <span className="truncate font-medium text-zinc-900">{task.title}</span>
    ),
    owner: (
      <span className="truncate text-zinc-500">
        {task.owner ? task.owner.name : "Autopilot"}
      </span>
    ),
    status: (
      <Badge variant={getStatusVariant(task.status)} size="small">
        {getStatusLabel(task.status)}
      </Badge>
    ),
    age: (
      <span className="whitespace-nowrap tabular-nums text-zinc-500">
        {formatElapsed(task)}
      </span>
    ),
    spend: (
      <span className="whitespace-nowrap tabular-nums text-zinc-500">
        {formatSpend(task.spend_total)}
      </span>
    ),
  };
}
