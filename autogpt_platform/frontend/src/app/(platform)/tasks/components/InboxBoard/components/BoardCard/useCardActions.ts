import { toast } from "@/components/molecules/Toast/use-toast";
import type { Task } from "../../../../data/types";
import { useTaskAction } from "../../../../data/useTasksData";
import { isCancellable } from "../../../../helpers";
import { useCardApprovals } from "../../../CardApprovals/useCardApprovals";
import { useInboxRow } from "../../../InboxRow/useInboxRow";

/** What an open card can do: answer what it waits on, cancel, open. */
export function useCardActions(task: Task) {
  const row = useInboxRow(task);
  const approvals = useCardApprovals(task.id, row.cards);
  const stop = useTaskAction();

  function cancel() {
    if (!window.confirm("Stop this task? The expert stops working on it."))
      return;
    stop.mutate(
      { id: task.id, action: "stop" },
      {
        onError: (error) =>
          toast({
            title: "Couldn't cancel the task",
            description: error instanceof Error ? error.message : undefined,
            variant: "destructive",
          }),
      },
    );
  }

  return {
    cards: row.cards,
    question: row.cards.length
      ? null
      : (task.ask?.question ?? task.blocker?.short ?? null),
    action: row.cards.length ? null : row.action,
    runAction: row.runAction,
    busy: row.busy || approvals.pending !== null || stop.isPending,
    approve: approvals.approve,
    decline: approvals.decline,
    canCancel: isCancellable(task.state),
    cancel,
  };
}
