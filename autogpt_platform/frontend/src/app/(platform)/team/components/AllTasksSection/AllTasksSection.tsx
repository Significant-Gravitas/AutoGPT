"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import { isOutputType } from "@/components/organisms/WorkOutputSheet/helpers";
import { WorkOutputSheet } from "@/components/organisms/WorkOutputSheet/WorkOutputSheet";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { ExpertTask } from "./helpers";
import { useAllTasksSection } from "./useAllTasksSection";

interface Props {
  experts: Expert[];
  enabled: boolean;
}

export function AllTasksSection({ experts, enabled }: Props) {
  const [activeTask, setActiveTask] = useState<ExpertTask | null>(null);
  const {
    tasks,
    reviewCount,
    needsReviewOnly,
    toggleNeedsReviewOnly,
    isLoading,
    isError,
    refetch,
  } = useAllTasksSection({ experts, enabled });

  if (experts.length === 0) {
    return (
      <Text variant="body" className="text-zinc-500">
        Hire an expert and their finished work will show up here.
      </Text>
    );
  }

  return (
    <section aria-label="All tasks">
      <div className="mb-2.5 flex items-center justify-end gap-2 empty:mb-0">
        {reviewCount > 0 ? (
          <button
            type="button"
            aria-pressed={needsReviewOnly}
            onClick={toggleNeedsReviewOnly}
            className={cn(
              "rounded-full px-3 py-1 text-xs font-medium ring-1 ring-inset transition-colors",
              needsReviewOnly
                ? "bg-amber-100 text-amber-700 ring-amber-200"
                : "bg-white text-zinc-500 ring-zinc-200 hover:text-zinc-800",
            )}
          >
            Needs review ({reviewCount})
          </button>
        ) : null}
      </div>

      {isLoading ? (
        <div className="space-y-3">
          <Skeleton className="h-16 w-full rounded-xl" />
          <Skeleton className="h-16 w-full rounded-xl" />
        </div>
      ) : isError ? (
        <ErrorCard
          context="your team's tasks"
          hint="We could not load what your experts have been working on."
          onRetry={refetch}
        />
      ) : tasks.length === 0 ? (
        <Text variant="body" className="text-zinc-500">
          {needsReviewOnly
            ? "Nothing is waiting on your review."
            : "No completed work yet. Finished runs will show up here."}
        </Text>
      ) : (
        <ul className="flex flex-col gap-3">
          {tasks.map((task) => (
            <li key={task.run.execution_id}>
              <TaskRow task={task} onOpen={() => setActiveTask(task)} />
            </li>
          ))}
        </ul>
      )}

      {activeTask ? (
        <WorkOutputSheet
          open
          onOpenChange={(open) => {
            if (!open) setActiveTask(null);
          }}
          title={activeTask.run.agent_name}
          outputType={
            isOutputType(activeTask.run.output_type)
              ? activeTask.run.output_type
              : "unknown"
          }
          outputKey={activeTask.run.output_key}
          graphId={activeTask.run.graph_id}
          executionId={activeTask.run.execution_id}
          runLink={activeTask.run.link}
        />
      ) : null}
    </section>
  );
}

function TaskRow({ task, onOpen }: { task: ExpertTask; onOpen: () => void }) {
  const { run, expert } = task;

  return (
    <div className="flex items-center justify-between gap-3 rounded-xl bg-white p-3 ring-1 ring-inset ring-zinc-200">
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-zinc-900">
          {run.agent_name}
        </p>
        <div className="mt-1 flex items-center gap-2">
          <Text variant="small" className="text-zinc-500">
            {expert.name}
          </Text>
          <RunStatusBadge status={run.status} />
          {run.needs_review && run.status.toUpperCase() !== "REVIEW" ? (
            <Badge variant="warning" size="small">
              Needs review
            </Badge>
          ) : null}
        </div>
      </div>
      <Button variant="secondary" size="small" onClick={onOpen}>
        Open
      </Button>
    </div>
  );
}
