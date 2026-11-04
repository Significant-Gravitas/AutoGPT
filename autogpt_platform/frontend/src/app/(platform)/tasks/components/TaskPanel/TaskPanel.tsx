"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ActivityCheckpoints } from "./components/ActivityCheckpoints/ActivityCheckpoints";
import { AttentionCard } from "./components/AttentionCard/AttentionCard";
import { CommentComposer } from "./components/CommentComposer/CommentComposer";
import { OriginalAsk } from "./components/OriginalAsk/OriginalAsk";
import { PanelSkeleton } from "./components/PanelSkeleton/PanelSkeleton";
import { Progress } from "./components/Progress/Progress";
import { ResultCard } from "./components/ResultCard/ResultCard";
import { ShowWork } from "./components/ShowWork/ShowWork";
import { SubtaskList } from "./components/SubtaskList/SubtaskList";
import { TaskHeader } from "./components/TaskHeader/TaskHeader";
import { useTaskPanel } from "./useTaskPanel";

interface Props {
  taskId: string;
  onOpenTask(id: string): void;
}

/** Top to bottom: the title, what was asked, one card (the outcome, what it
 *  needs from you, or why it failed), the feed, and Show work. */
export function TaskPanel({ taskId, onOpenTask }: Props) {
  const panel = useTaskPanel(taskId);
  const { task, view } = panel;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div
        key={taskId}
        className="flex min-h-0 flex-1 animate-inbox-enter flex-col"
      >
        {panel.isLoading || (!task && !panel.isError) ? (
          <PanelSkeleton />
        ) : panel.isError || !task || !view ? (
          <div className="p-4">
            <ErrorCard
              context="task"
              httpError={{ message: "This task could not be loaded." }}
              onRetry={panel.retry}
            />
          </div>
        ) : (
          <>
            <div className="min-h-0 flex-1 divide-y divide-border-hairline overflow-y-auto">
              <div className="flex flex-col gap-3 px-1.5 py-5">
                <TaskHeader
                  task={task}
                  experts={panel.experts}
                  onUpdate={panel.update}
                />
                <OriginalAsk
                  title={task.title}
                  description={task.description}
                />
              </div>
              {view.card === "outcome" ? (
                <ResultCard
                  taskId={task.id}
                  summary={view.resultSummary}
                  evidence={view.evidence}
                />
              ) : null}
              {view.card === "attention" ? (
                <div className="px-1.5 py-4">
                  <AttentionCard
                    key={
                      view.ask?.id ?? view.blocker?.what_stopped_it ?? "wait"
                    }
                    taskId={task.id}
                    goal={view.contract?.goal ?? task.title}
                    askerName={task.assignee?.name ?? "Otto"}
                    failed={view.failed}
                    ask={view.ask}
                    blocker={view.blocker}
                    approvals={view.approvals}
                    awaitingAccept={view.awaitingAccept}
                    resultSummary={view.resultSummary}
                    unmet={view.unmet}
                    proof={view.awaitingAccept ? view.evidence : []}
                    workChatHref={view.workChatHref}
                  />
                </div>
              ) : null}
              {view.contract && view.state === "working" ? (
                <Progress
                  plan={view.contract.plan ?? []}
                  now={view.now}
                  state={view.state}
                />
              ) : null}
              <ActivityCheckpoints
                taskId={task.id}
                title={task.title}
                running={view.running}
                resultSummary={view.resultSummary}
              />
              <SubtaskList task={task} onOpenTask={onOpenTask} />
              <ShowWork
                task={task}
                contract={view.contract}
                state={view.state}
                now={view.now}
                running={view.running}
                proofInCard={view.proofInCard}
              />
            </div>
            <div className="shrink-0 border-t border-border-hairline px-1.5 pt-3 pb-1">
              <CommentComposer taskId={task.id} experts={panel.experts} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
