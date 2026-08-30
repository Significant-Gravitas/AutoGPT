"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { TaskRunRef } from "@/app/api/__generated__/models/taskRunRef";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import Link from "next/link";
import {
  formatElapsed,
  formatSpend,
  getStatusLabel,
  getStatusVariant,
  isOpenTask,
} from "../../task-helpers";
import { useTaskDetailDrawer } from "./useTaskDetailDrawer";

interface Props {
  taskId: string | null;
  onClose: () => void;
}

export function TaskDetailDrawer({ taskId, onClose }: Props) {
  const { task, children, isLoading, isError, cancel, isCancelling } =
    useTaskDetailDrawer({ taskId });

  return (
    <Sheet
      open={taskId !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent
        side="right"
        className="flex w-full flex-col gap-4 overflow-y-auto sm:max-w-xl"
      >
        <SheetHeader className="text-left">
          <SheetTitle className="truncate">
            {task ? task.title : "Task"}
          </SheetTitle>
        </SheetHeader>

        {isLoading ? (
          <div className="space-y-3" data-testid="task-detail-loading">
            <Skeleton className="h-6 w-2/3" />
            <Skeleton className="h-40 w-full" />
          </div>
        ) : isError || !task ? (
          <ErrorCard
            context="this task"
            hint="We could not load this task."
            onRetry={onClose}
          />
        ) : (
          <TaskDetailBody
            task={task}
            childTasks={children}
            onCancel={cancel}
            isCancelling={isCancelling}
          />
        )}
      </SheetContent>
    </Sheet>
  );
}

interface BodyProps {
  task: DelegatedTask;
  childTasks: DelegatedTask[];
  onCancel: () => void;
  isCancelling: boolean;
}

function TaskDetailBody({
  task,
  childTasks,
  onCancel,
  isCancelling,
}: BodyProps) {
  const runs = task.runs ?? [];

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={getStatusVariant(task.status)} size="small">
          {getStatusLabel(task.status)}
        </Badge>
        <Text variant="small" className="text-zinc-500">
          {formatElapsed(task)}
        </Text>
        <Text variant="small" className="text-zinc-500">
          {formatSpend(task.spend_total)} spent
        </Text>
        {task.owner ? (
          <Text variant="small" className="text-zinc-500">
            {task.owner.name}
          </Text>
        ) : null}
      </div>

      <Section title="What was asked for">
        <p className="whitespace-pre-line text-sm text-zinc-600">{task.spec}</p>
      </Section>

      {task.outcome_summary ? (
        <Section title="Outcome">
          <p className="whitespace-pre-line text-sm text-zinc-600">
            {task.outcome_summary}
          </p>
        </Section>
      ) : null}

      {runs.length > 0 ? (
        <Section title="Linked runs">
          <ul className="flex flex-col gap-2" aria-label="Linked runs">
            {runs.map((run) => (
              <li key={run.execution_id}>
                <LinkedRunRow run={run} />
              </li>
            ))}
          </ul>
        </Section>
      ) : null}

      {childTasks.length > 0 ? (
        <Section title="Follow-on tasks">
          <ul className="flex flex-col gap-2" aria-label="Follow-on tasks">
            {childTasks.map((child) => (
              <li
                key={child.id}
                className="flex items-center gap-2 rounded-xl bg-zinc-50 p-2"
              >
                <Badge variant={getStatusVariant(child.status)} size="small">
                  {getStatusLabel(child.status)}
                </Badge>
                <span className="truncate text-sm text-zinc-700">
                  {child.title}
                </span>
              </li>
            ))}
          </ul>
        </Section>
      ) : null}

      {isOpenTask(task) ? (
        <Button
          variant="destructive"
          size="small"
          className="self-start"
          loading={isCancelling}
          onClick={onCancel}
        >
          Cancel task
        </Button>
      ) : null}
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-2">
      <Text variant="small" className="font-medium text-zinc-900">
        {title}
      </Text>
      {children}
    </section>
  );
}

function LinkedRunRow({ run }: { run: TaskRunRef }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl bg-white p-3 ring-1 ring-inset ring-zinc-200">
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-zinc-900">
          {run.agent_name}
        </p>
        <div className="mt-1">
          <RunStatusBadge status={run.status} />
        </div>
      </div>
      {run.link ? (
        <Link
          href={run.link}
          className="whitespace-nowrap text-sm text-zinc-500 hover:text-zinc-800"
        >
          Open run
        </Link>
      ) : null}
    </div>
  );
}
