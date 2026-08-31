"use client";

import Link from "next/link";
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  getGetTaskQueryKey,
  getListTasksQueryKey,
  useAcceptTask,
  useRejectTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { toast } from "@/components/molecules/Toast/use-toast";
import { cn } from "@/lib/utils";
import { CARD } from "./ResultCards";
import { str } from "./resultHelpers";

interface Props {
  output: Record<string, unknown>;
}

const ACTION_LABELS: Record<string, string> = {
  handoff: "Handed off",
  escalation: "Escalated to you",
  report: "Outcome reported",
};

/** An expert-side task action (report / escalate / handoff) surfaced in the
 *  chain: what happened, the task it touched, and the door into its Team
 *  detail. A reported outcome also gets Accept / Request-changes controls. */
export function TaskUpdateCard({ output }: Props) {
  const taskId = str(output, "task_id");
  const action = str(output, "action") ?? "report";
  const title = str(output, "task_title");
  if (!taskId) return null;

  return (
    <div className={cn(CARD, "w-full rounded-2xl p-2.5")}>
      <div className="flex items-center gap-2.5">
        <div className="min-w-0 flex-1">
          <p className="truncate text-[13px] font-medium text-zinc-800">
            {ACTION_LABELS[action] ?? "Task updated"}
          </p>
          <p className="truncate text-xs text-zinc-500">{title ?? "Task"}</p>
        </div>
        <Link
          href={`/team/tasks/${encodeURIComponent(taskId)}`}
          className="shrink-0 text-xs text-zinc-500 transition-colors hover:text-zinc-800"
        >
          View task
        </Link>
      </div>
      {action === "report" ? <ReportReview taskId={taskId} /> : null}
    </div>
  );
}

function ReportReview({ taskId }: { taskId: string }) {
  const queryClient = useQueryClient();
  const [isReviewed, setIsReviewed] = useState(false);
  const [isRequestingChanges, setIsRequestingChanges] = useState(false);
  const [note, setNote] = useState("");

  async function refreshTask() {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: getGetTaskQueryKey(taskId) }),
      queryClient.invalidateQueries({ queryKey: getListTasksQueryKey() }),
    ]);
  }

  const { mutate: accept, isPending: isAccepting } = useAcceptTask({
    mutation: {
      onSuccess: async (res) => {
        toast({ title: okData(res)?.message ?? "Outcome accepted" });
        setIsReviewed(true);
        await refreshTask();
      },
      onError: () =>
        toast({
          title: "Could not accept the outcome",
          description: "Please try again.",
          variant: "destructive",
        }),
    },
  });

  const { mutate: reject, isPending: isRejecting } = useRejectTask({
    mutation: {
      onSuccess: async (res) => {
        toast({ title: okData(res)?.message ?? "Changes requested" });
        setIsReviewed(true);
        await refreshTask();
      },
      onError: () =>
        toast({
          title: "Could not send your changes",
          description: "Please try again.",
          variant: "destructive",
        }),
    },
  });

  function submitChanges() {
    const trimmed = note.trim();
    if (!trimmed || isRejecting) return;
    reject({ taskId, data: { note: trimmed } });
  }

  if (isReviewed) return null;

  if (isRequestingChanges) {
    return (
      <form
        className="mt-2 flex flex-col gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          submitChanges();
        }}
      >
        <Input
          id={`task-update-note-${taskId}`}
          label="What should change?"
          hideLabel
          size="small"
          placeholder="What should change?"
          value={note}
          onChange={(event) => setNote(event.target.value)}
          disabled={isRejecting}
          wrapperClassName="mb-0"
        />
        <div className="flex items-center gap-2">
          <Button
            type="submit"
            variant="primary"
            size="small"
            loading={isRejecting}
            disabled={!note.trim()}
          >
            Send changes
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="small"
            disabled={isRejecting}
            onClick={() => {
              setIsRequestingChanges(false);
              setNote("");
            }}
          >
            Cancel
          </Button>
        </div>
      </form>
    );
  }

  return (
    <div className="mt-2 flex items-center gap-2">
      <Button
        variant="primary"
        size="small"
        loading={isAccepting}
        onClick={() => accept({ taskId })}
      >
        Accept
      </Button>
      <Button
        variant="secondary"
        size="small"
        disabled={isAccepting}
        onClick={() => setIsRequestingChanges(true)}
      >
        Request changes
      </Button>
    </div>
  );
}
