"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  Sheet,
  SheetContent,
  SheetTitle,
} from "@/components/molecules/Sheet/Sheet";
import { ActivityFeed } from "./components/ActivityFeed/ActivityFeed";
import { InboxStatus } from "./components/InboxStatus/InboxStatus";
import { CommentComposer } from "./components/CommentComposer/CommentComposer";
import { LiveStrip } from "./components/LiveStrip/LiveStrip";
import { QuestionCard } from "./components/QuestionCard/QuestionCard";
import { SheetSection } from "./components/SheetSection/SheetSection";
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
        className="flex w-full flex-col gap-1.5 bg-surface-tray p-1.5 data-[state=open]:ease-out-expo sm:max-w-xl"
      >
        <SheetTitle className="sr-only">{task?.title ?? "Task"}</SheetTitle>
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-border-card bg-surface shadow-sm">
          {isLoading || (!task && !isError) ? (
            <SheetSkeleton />
          ) : isError || !task ? (
            <div className="px-5 py-5">
              <Text variant="body" tone="quiet">
                This task could not be loaded.
              </Text>
            </div>
          ) : (
            <div className="flex-1 divide-y divide-border-hairline overflow-y-auto">
              <header className="flex flex-col gap-4 py-5 pr-12 pl-5">
                <TaskHeader task={task} experts={experts} onUpdate={update} />
                <InboxStatus task={task} />
              </header>
              {task.open_question ? (
                <div className="px-5 py-4">
                  <QuestionCard
                    taskId={task.id}
                    question={task.open_question}
                  />
                </div>
              ) : null}
              {task.status === "in_progress" && task.live ? (
                <div className="px-5 py-4">
                  <LiveStrip live={task.live} metrics={task.metrics} />
                </div>
              ) : null}
              <SheetSection label="Description">
                {task.description ? (
                  <Text
                    variant="body-tight"
                    tone="text"
                    className="whitespace-pre-wrap"
                  >
                    {task.description}
                  </Text>
                ) : (
                  <Text variant="body-tight" tone="faint">
                    No description
                  </Text>
                )}
              </SheetSection>
              {task.status === "done" ? <TaskOutcome task={task} /> : null}
              <SubtaskList task={task} onOpenTask={onOpenTask} />
              <ActivityFeed
                items={activity}
                isLoading={isActivityLoading}
                experts={experts}
              />
            </div>
          )}
        </div>
        {task ? (
          <div className="rounded-2xl border border-border-card bg-surface p-3 shadow-sm">
            <CommentComposer taskId={task.id} experts={experts} />
          </div>
        ) : null}
      </SheetContent>
    </Sheet>
  );
}

function SheetSkeleton() {
  return (
    <div className="flex flex-col gap-4 px-5 py-5">
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
