"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ActivityCheckpoints } from "./components/ActivityCheckpoints/ActivityCheckpoints";
import { AttentionCard } from "./components/AttentionCard/AttentionCard";
import { unmetChecks } from "./components/AttentionCard/helpers";
import { CommentComposer } from "./components/CommentComposer/CommentComposer";
import { DoneWhen } from "./components/DoneWhen/DoneWhen";
import { GoalLine } from "./components/GoalLine/GoalLine";
import { PanelApprovals } from "./components/PanelApprovals/PanelApprovals";
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

export function TaskPanel({ taskId, onOpenTask }: Props) {
  const { task, summary, isLoading, isError, retry, experts, update } =
    useTaskPanel(taskId);
  const contract = summary?.contract;
  const ask = contract?.ask ?? task?.ask ?? null;
  const blocker = contract?.blocker ?? task?.blocker ?? null;
  const rawStatus = summary?.raw_status ?? task?.raw_status;
  const state = summary?.state ?? task?.state;
  const settled = state === "done" || state === "stopped";
  const cards = settled ? [] : (contract?.approvals ?? task?.approvals ?? []);

  return (
    <div className="flex h-full min-h-0 flex-col">
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
              <div className="flex flex-col gap-3 px-1.5 py-5">
                <TaskHeader task={task} experts={experts} onUpdate={update} />
                {contract ? (
                  <GoalLine
                    title={task.title}
                    goal={contract.goal}
                    previousGoal={contract.previous_goal}
                  />
                ) : null}
              </div>
              {cards.length && task ? (
                <div className="px-1.5 py-4">
                  <PanelApprovals
                    taskId={task.id}
                    askerName={task.assignee?.name ?? "Otto"}
                    cards={cards}
                  />
                </div>
              ) : !settled &&
                (ask || blocker || rawStatus === "awaiting_accept") ? (
                <div className="px-1.5 py-4">
                  <AttentionCard
                    key={ask?.id ?? blocker?.what_stopped_it ?? "accept"}
                    taskId={task.id}
                    goal={contract?.goal ?? task.title}
                    askerName={task.assignee?.name ?? "Otto"}
                    ask={ask}
                    blocker={blocker}
                    awaitingAccept={rawStatus === "awaiting_accept"}
                    resultSummary={contract?.result?.summary}
                    unmet={unmetChecks(contract?.done_when)}
                  />
                </div>
              ) : null}
              {contract ? (
                <DoneWhen taskId={task.id} contract={contract} />
              ) : null}
              {contract && task.state !== "done" ? (
                <Progress
                  plan={contract.plan ?? []}
                  now={summary?.now ?? task.now}
                  state={summary?.state ?? task.state}
                />
              ) : null}
              {contract && task.state === "done" ? (
                <ResultCard taskId={task.id} contract={contract} />
              ) : null}
              <SubtaskList task={task} onOpenTask={onOpenTask} />
              <ActivityCheckpoints
                taskId={task.id}
                title={task.title}
                running={summary?.live?.running ?? task.state === "working"}
              />
              <ShowWork task={task} experts={experts} />
            </div>
            <div className="shrink-0 border-t border-border-hairline px-1.5 pt-3 pb-1">
              <CommentComposer taskId={task.id} experts={experts} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
