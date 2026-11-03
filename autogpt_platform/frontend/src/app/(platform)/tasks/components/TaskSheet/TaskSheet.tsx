"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  Sheet,
  SheetContent,
  SheetTitle,
} from "@/components/molecules/Sheet/Sheet";
import { ActivityFeed } from "./components/ActivityFeed/ActivityFeed";
import { CommentComposer } from "./components/CommentComposer/CommentComposer";
import { LiveStrip } from "./components/LiveStrip/LiveStrip";
import { QuestionCard } from "./components/QuestionCard/QuestionCard";
import { SubtaskList } from "./components/SubtaskList/SubtaskList";
import { TaskHeader } from "./components/TaskHeader/TaskHeader";
import { TaskOutcome } from "./components/TaskOutcome/TaskOutcome";
import { useTaskSheet } from "./useTaskSheet";

interface Props {
  taskId: string | null;
  onOpenChange: (open: boolean) => void;
  onOpenTask?: (taskId: string) => void;
}

export function TaskSheet({ taskId, onOpenChange, onOpenTask }: Props) {
  const {
    task,
    isLoading,
    isError,
    activity,
    isActivityLoading,
    experts,
    update,
  } = useTaskSheet(taskId);

  return (
    <Sheet open={!!taskId} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        aria-describedby={undefined}
        className="flex w-full flex-col sm:max-w-xl"
      >
        <SheetTitle className="sr-only">{task?.title ?? "Task"}</SheetTitle>
        {isLoading || (!task && !isError) ? (
          <SheetSkeleton />
        ) : isError || !task ? (
          <div>
            <Text variant="body" tone="muted">
              This task could not be loaded.
            </Text>
          </div>
        ) : (
          <>
            <div className="flex-1 overflow-y-auto">
              <div className="flex flex-col gap-6 pb-4">
                <TaskHeader task={task} experts={experts} onUpdate={update} />
                {task.open_question ? (
                  <QuestionCard
                    taskId={task.id}
                    question={task.open_question}
                  />
                ) : null}
                {task.status === "in_progress" && task.live ? (
                  <LiveStrip live={task.live} metrics={task.metrics} />
                ) : null}
                {task.description ? (
                  <Text
                    variant="body-tight"
                    tone="body"
                    className="whitespace-pre-wrap"
                  >
                    {task.description}
                  </Text>
                ) : (
                  <Text variant="body-tight" tone="subtle">
                    No description
                  </Text>
                )}
                {task.status === "done" ? <TaskOutcome task={task} /> : null}
                <SubtaskList task={task} onOpenTask={onOpenTask} />
                <ActivityFeed
                  items={activity}
                  isLoading={isActivityLoading}
                  experts={experts}
                />
              </div>
            </div>
            <CommentComposer taskId={task.id} experts={experts} />
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}

function SheetSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      <Skeleton className="h-4 w-16" />
      <Skeleton className="h-7 w-3/4" />
      <div className="flex gap-2">
        <Skeleton className="h-7 w-24" />
        <Skeleton className="h-7 w-24" />
        <Skeleton className="h-7 w-24" />
      </div>
      <Skeleton className="h-20 w-full" />
      <Skeleton className="h-32 w-full" />
    </div>
  );
}
