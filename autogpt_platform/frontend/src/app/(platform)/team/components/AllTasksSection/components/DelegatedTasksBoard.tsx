"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { TaskDetailDrawer } from "../../TaskDetailDrawer/TaskDetailDrawer";
import { DelegatedTaskRow } from "./DelegatedTaskRow";
import { useDelegatedTasksBoard } from "./useDelegatedTasksBoard";

interface Props {
  enabled: boolean;
}

/** The task-spine version of the team board: every DelegatedTask receipt
 *  across the whole team — including Autopilot work, which the run-based
 *  board can't see because it fans out per hired expert. */
export function DelegatedTasksBoard({ enabled }: Props) {
  const {
    activeTasks,
    historyTasks,
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

  if (activeTasks.length === 0 && historyTasks.length === 0) {
    return (
      <Text variant="body" className="text-zinc-500">
        Nothing delegated yet. Ask an expert (or Autopilot) to do something and
        it will show up here.
      </Text>
    );
  }

  return (
    <section aria-label="All tasks" className="flex flex-col gap-6">
      <TaskGroup
        title="Active"
        tasks={activeTasks}
        emptyText="Nothing in flight right now."
        onOpen={openTask}
      />
      <TaskGroup
        title="History"
        tasks={historyTasks}
        emptyText="No finished tasks yet."
        onOpen={openTask}
      />

      <TaskDetailDrawer taskId={openTaskId} onClose={closeTask} />
    </section>
  );
}

interface GroupProps {
  title: string;
  tasks: DelegatedTask[];
  emptyText: string;
  onOpen: (taskId: string) => void;
}

function TaskGroup({ title, tasks, emptyText, onOpen }: GroupProps) {
  return (
    <div className="flex flex-col gap-2.5">
      <Text variant="small" className="font-medium text-zinc-900">
        {title}
      </Text>
      {tasks.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          {emptyText}
        </Text>
      ) : (
        <ul className="flex flex-col gap-3" aria-label={`${title} tasks`}>
          {tasks.map((task) => (
            <li key={task.id}>
              <DelegatedTaskRow task={task} onOpen={() => onOpen(task.id)} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
