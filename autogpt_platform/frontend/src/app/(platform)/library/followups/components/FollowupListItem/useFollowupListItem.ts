import { useDeleteV1DeleteExecutionSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import { useQueryClient } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import { useState } from "react";
import { describeFollowup, formatNextRunTitle } from "./helpers";

interface Args {
  followup: CopilotTurnJobInfo;
}

export function useFollowupListItem({ followup }: Args) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isViewOpen, setIsViewOpen] = useState(false);

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
  const nextRunTitle = formatNextRunTitle(
    followup.next_run_time,
    followup.user_timezone ?? followup.timezone,
  );

  const recurrenceLabel = followup.cron
    ? safeHumanizeCron(followup.cron)
    : "Runs once";

  function openDelete() {
    setIsDeleteOpen(true);
  }

  function closeDelete(open: boolean) {
    setIsDeleteOpen(open);
  }

  function openView() {
    setIsViewOpen(true);
  }

  function closeView(open: boolean) {
    setIsViewOpen(open);
  }

  async function handleDelete() {
    try {
      await deleteSchedule({ scheduleId: followup.id });
      toast({ title: "Follow-up deleted" });
      setIsDeleteOpen(false);
      invalidateAllScheduleQueries(queryClient);
    } catch (error) {
      toast({
        title: "Failed to delete follow-up",
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
    isViewOpen,
    openView,
    closeView,
    fullMessage: followup.message || "(no message)",
  };
}

function safeHumanizeCron(cron: string): string {
  try {
    return humanizeCronExpression(cron);
  } catch {
    return cron;
  }
}
