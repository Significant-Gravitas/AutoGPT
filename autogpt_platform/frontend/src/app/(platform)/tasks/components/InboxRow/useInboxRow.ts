import { toast } from "@/components/molecules/Toast/use-toast";
import { useNowMinute } from "@/hooks/useNowMinute";
import { useRouter } from "next/navigation";
import { cardAction, shortAgo } from "../../data/helpers";
import type { Task } from "../../data/types";
import { useAnswerAsk, useTaskAction } from "../../data/useTasksData";

function showError(title: string) {
  return function onError(error: unknown) {
    toast({
      title,
      description: error instanceof Error ? error.message : undefined,
      variant: "destructive",
    });
  };
}

export function useInboxRow(task: Task) {
  useNowMinute();
  const router = useRouter();
  const answer = useAnswerAsk();
  const act = useTaskAction();
  const action = cardAction(task);

  function runAction() {
    if (!action) return;
    if (action.kind === "answer") {
      answer.mutate(
        { id: task.id, answer: action.label, askId: action.askId },
        { onError: showError("Couldn't send your answer") },
      );
      return;
    }
    act.mutate(
      { id: task.id, action: action.action },
      {
        onSuccess: (summary) => {
          if (action.action === "do_in_chat" && summary.work_session_id) {
            router.push(`/copilot?sessionId=${summary.work_session_id}`);
          }
        },
        onError: showError("Couldn't do that"),
      },
    );
  }

  const open = task.state !== "done" && task.state !== "stopped";

  return {
    cards: open ? (task.approvals ?? []) : [],
    ago: shortAgo(task.updated_at),
    action,
    busy: answer.isPending || act.isPending,
    runAction,
  };
}
