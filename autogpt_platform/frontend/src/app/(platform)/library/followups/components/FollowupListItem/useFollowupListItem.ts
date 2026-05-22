import {
  getListCopilotFollowupSchedulesQueryKey,
  useDeleteV1DeleteExecutionSchedule,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { useQueryClient } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import { useState } from "react";
import { describeFollowup } from "./helpers";

interface Args {
  followup: CopilotTurnJobInfo;
  onDeleted?: () => void;
}

export function useFollowupListItem({ followup, onDeleted }: Args) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);

  const { mutateAsync: deleteSchedule, isPending: isDeleting } =
    useDeleteV1DeleteExecutionSchedule();

  const { messagePreview, sessionHref } = describeFollowup(followup);

  const nextRunDate = followup.next_run_time
    ? new Date(followup.next_run_time)
    : null;
  const nextRunLabel =
    nextRunDate && !Number.isNaN(nextRunDate.valueOf())
      ? `Next ${formatDistanceToNow(nextRunDate, { addSuffix: true })}`
      : "Pending";
  const nextRunTitle = nextRunDate ? nextRunDate.toString() : undefined;

  const recurrenceLabel = followup.cron
    ? safeHumanizeCron(followup.cron)
    : "Runs once";

  function openDelete() {
    setIsDeleteOpen(true);
  }

  function closeDelete(open: boolean) {
    setIsDeleteOpen(open);
  }

  async function handleDelete() {
    try {
      await deleteSchedule({ scheduleId: followup.id });
      toast({ title: "Follow-up cancelled" });
      setIsDeleteOpen(false);
      onDeleted?.();
      queryClient.invalidateQueries({
        queryKey: getListCopilotFollowupSchedulesQueryKey(),
      });
    } catch (error) {
      toast({
        title: "Failed to cancel follow-up",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  return {
    sessionHref,
    messagePreview,
    nextRunLabel,
    nextRunTitle,
    recurrenceLabel,
    isDeleteOpen,
    openDelete,
    closeDelete,
    isDeleting,
    handleDelete,
  };
}

function safeHumanizeCron(cron: string): string {
  try {
    return humanizeCronExpression(cron);
  } catch {
    return cron;
  }
}
