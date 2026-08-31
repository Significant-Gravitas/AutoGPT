"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { notFound, useParams } from "next/navigation";
import Link from "next/link";
import { TaskActivity } from "./components/TaskActivity";
import { TaskOutcomeReview } from "./components/TaskOutcomeReview/TaskOutcomeReview";
import { TaskProperties } from "./components/TaskProperties";
import { TaskSpec } from "./components/TaskSpec";
import { TaskSubtasks } from "./components/TaskSubtasks";
import { TaskTopBar } from "./components/TaskTopBar";
import { useTaskDetailPage } from "./useTaskDetailPage";

// The top bar's hairline runs edge to edge, so the page holds no padding of
// its own — each band below it carries the gutters instead.
const MAIN_CLASS = "min-h-screen w-full pb-20";
const BAND_CLASS = "px-6 md:px-8";

export default function TaskDetailPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const { task, children, isLoading, isError, refetch, cancel, isCancelling } =
    useTaskDetailPage({ taskId, enabled: Boolean(enabled) && ready });

  if (!ready || isLoading) {
    return (
      <main className={MAIN_CLASS}>
        <div className={`${BAND_CLASS} pt-6`}>
          <Skeleton className="h-8 w-full rounded-xl" />
          <Skeleton className="mt-8 h-48 w-full rounded-2xl" />
        </div>
      </main>
    );
  }

  if (!enabled) {
    notFound();
  }

  if (isError || !task) {
    return (
      <main className={`${MAIN_CLASS} ${BAND_CLASS} pt-6`}>
        <Link
          href="/team"
          className="text-sm text-zinc-500 hover:text-zinc-800"
          data-testid="task-back-to-team"
        >
          Team
        </Link>
        <div className="mt-6">
          <ErrorCard
            context="this task"
            hint="We could not load this task."
            onRetry={refetch}
          />
        </div>
      </main>
    );
  }

  return (
    <main className={MAIN_CLASS}>
      <TaskTopBar task={task} />

      <div
        className={`${BAND_CLASS} mt-8 grid gap-10 lg:grid-cols-[minmax(0,1fr)_20rem]`}
      >
        <div className="min-w-0 space-y-8">
          <div>
            <h1 className="text-[22px] font-semibold leading-snug tracking-[-0.015em] text-zinc-900">
              {task.title}
            </h1>
            <TaskSpec spec={task.spec} />
          </div>

          {task.outcome_summary ? (
            <section className="flex flex-col gap-2 rounded-2xl bg-zinc-50 p-4">
              <Text variant="small" className="font-medium text-zinc-900">
                Outcome
              </Text>
              <p className="whitespace-pre-line text-sm leading-6 text-zinc-600">
                {task.outcome_summary}
              </p>
              <TaskOutcomeReview task={task} />
            </section>
          ) : null}

          <TaskSubtasks taskId={task.id} childTasks={children} />
          <TaskActivity task={task} />
        </div>

        <TaskProperties
          task={task}
          onCancel={cancel}
          isCancelling={isCancelling}
        />
      </div>
    </main>
  );
}
