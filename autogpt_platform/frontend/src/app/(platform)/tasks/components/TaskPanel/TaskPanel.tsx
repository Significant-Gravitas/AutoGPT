"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ActivityFeed } from "./components/ActivityFeed/ActivityFeed";
import { CommentComposer } from "./components/CommentComposer/CommentComposer";
import { InboxStatus } from "./components/InboxStatus/InboxStatus";
import { LiveStrip } from "./components/LiveStrip/LiveStrip";
import { PanelHeader } from "./components/PanelHeader/PanelHeader";
import { PanelPlan } from "./components/PanelPlan/PanelPlan";
import { PanelSection } from "./components/PanelSection/PanelSection";
import { PanelSkeleton } from "./components/PanelSkeleton/PanelSkeleton";
import { QuestionCard } from "./components/QuestionCard/QuestionCard";
import { SubtaskList } from "./components/SubtaskList/SubtaskList";
import { TaskHeader } from "./components/TaskHeader/TaskHeader";
import { TaskOutcome } from "./components/TaskOutcome/TaskOutcome";
import { useTaskPanel } from "./useTaskPanel";

interface Props {
  taskId: string;
  onClose(): void;
  onOpenTask(id: string): void;
}

export function TaskPanel({ taskId, onClose, onOpenTask }: Props) {
  const {
    task,
    isLoading,
    isError,
    retry,
    activity,
    isActivityLoading,
    experts,
    update,
  } = useTaskPanel(taskId);

  return (
    <div className="flex h-full min-h-0 flex-col bg-surface">
      <PanelHeader task={task} onClose={onClose} />
      <div
        key={taskId}
        className="flex min-h-0 flex-1 animate-inbox-enter flex-col"
      >
        {isLoading || (!task && !isError) ? (
          <PanelSkeleton />
        ) : isError || !task ? (
          <div className="p-4">
            <ErrorCard
              context="task"
              httpError={{ message: "This task could not be loaded." }}
              onRetry={retry}
            />
          </div>
        ) : (
          <>
            <div className="min-h-0 flex-1 divide-y divide-border-hairline overflow-y-auto">
              <div className="flex flex-col gap-4 px-5 py-5">
                <TaskHeader task={task} experts={experts} onUpdate={update} />
                <InboxStatus task={task} />
              </div>
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
              {task.plan_id ? <PanelPlan planId={task.plan_id} /> : null}
              <PanelSection label="Description">
                {task.description ? (
                  <Text
                    variant="inbox"
                    tone="text"
                    className="whitespace-pre-wrap"
                  >
                    {task.description}
                  </Text>
                ) : (
                  <Text variant="inbox" tone="faint">
                    No description
                  </Text>
                )}
              </PanelSection>
              {task.status === "done" ? <TaskOutcome task={task} /> : null}
              <SubtaskList task={task} onOpenTask={onOpenTask} />
              <ActivityFeed
                items={activity}
                isLoading={isActivityLoading}
                experts={experts}
              />
            </div>
            <div className="shrink-0 border-t border-border-hairline bg-surface px-4 pt-3 pb-4">
              <CommentComposer taskId={task.id} experts={experts} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
