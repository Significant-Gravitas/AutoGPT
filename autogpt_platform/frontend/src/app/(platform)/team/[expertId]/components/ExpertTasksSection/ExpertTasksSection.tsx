"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { TaskRow } from "./components/TaskRow";
import { useExpertTasksSection } from "./useExpertTasksSection";

interface Props {
  expertId: string;
  enabled: boolean;
}

export function ExpertTasksSection({ expertId, enabled }: Props) {
  const { activeTasks, historyTasks, isLoading, isError, refetch } =
    useExpertTasksSection({ expertId, enabled });

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
        context="this expert's tasks"
        hint="We could not load what this expert has been asked to do."
        onRetry={refetch}
      />
    );
  }

  if (activeTasks.length === 0 && historyTasks.length === 0) {
    return (
      <Text variant="body" className="text-zinc-500">
        Nothing delegated yet. Ask this expert to do something and it will show
        up here.
      </Text>
    );
  }

  return (
    <section className="flex flex-col gap-6">
      <TaskGroup
        title="Active"
        tasks={activeTasks}
        emptyText="Nothing in flight right now."
      />
      <TaskGroup
        title="History"
        tasks={historyTasks}
        emptyText="No finished tasks yet."
      />
    </section>
  );
}

interface GroupProps {
  title: string;
  tasks: DelegatedTask[];
  emptyText: string;
}

function TaskGroup({ title, tasks, emptyText }: GroupProps) {
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
              <TaskRow task={task} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
