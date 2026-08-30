"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { FilterTable } from "@/components/molecules/FilterTable/FilterTable";
import {
  Clock01Icon,
  DollarCircleIcon,
  Progress02Icon,
  Task01Icon,
  TaskDone01Icon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import { getExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import {
  TASK_FILTER_ALL_ICON,
  TASK_TABLE_FILTERS,
  formatElapsed,
  formatSpend,
  getTaskFilterKey,
  getTaskOrbVariant,
} from "../../../task-helpers";
import { TaskDetailDrawer } from "../../TaskDetailDrawer/TaskDetailDrawer";
import { TaskStatusChip } from "../../TaskStatusChip/TaskStatusChip";
import { useDelegatedTasksBoard } from "./useDelegatedTasksBoard";

interface Props {
  enabled: boolean;
}

const COLUMNS = [
  {
    key: "task",
    label: "Task name",
    icon: Task01Icon,
    width: "minmax(0,1.4fr)",
  },
  { key: "owner", label: "Owner", icon: UserIcon, width: "minmax(0,0.9fr)" },
  {
    key: "status",
    label: "Status",
    icon: Progress02Icon,
    width: "minmax(0,0.8fr)",
  },
  { key: "age", label: "When", icon: Clock01Icon, width: "minmax(0,0.6fr)" },
  {
    key: "spend",
    label: "Spend",
    icon: DollarCircleIcon,
    width: "minmax(0,0.5fr)",
  },
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

  return (
    <section aria-label="All tasks">
      <FilterTable
        ariaLabel="Delegated tasks"
        columns={COLUMNS}
        filters={TASK_TABLE_FILTERS}
        allIcon={TASK_FILTER_ALL_ICON}
        maxVisibleFilters={3}
        emptyIcon={TaskDone01Icon}
        emptyText="Nothing delegated yet. Ask an expert (or Autopilot) to do something and it will show up here."
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

/** A null owner means Autopilot did the work itself, which the copilot thread
 *  header also marks with the AutoGPT logo rather than a generated avatar. */
function OwnerCell({ owner }: { owner: DelegatedTask["owner"] }) {
  return (
    <span className="flex min-w-0 items-center gap-2">
      {owner ? (
        <ExpertAvatar name={owner.name} avatarUrl={owner.avatar_url} size={20} />
      ) : (
        <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-zinc-100 ring-1 ring-inset ring-zinc-200">
          <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-3" />
        </span>
      )}
      <span className="truncate text-zinc-500">
        {owner ? owner.name : "Autopilot"}
      </span>
    </span>
  );
}

function buildCells(task: DelegatedTask) {
  return {
    task: (
      <span className="truncate font-medium text-zinc-900">{task.title}</span>
    ),
    owner: <OwnerCell owner={task.owner} />,
    status: (
      <TaskStatusChip
        status={task.status}
        orbVariant={getTaskOrbVariant(task.id)}
        accentClassName={
          task.owner ? getExpertAccent(task.owner.role).icon : undefined
        }
      />
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
