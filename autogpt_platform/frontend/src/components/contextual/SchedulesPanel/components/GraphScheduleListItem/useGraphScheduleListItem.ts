import { useDeleteV1DeleteExecutionSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import { useQueryClient } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import { useState } from "react";

interface Args {
  schedule: GraphExecutionJobInfo;
}

export function useGraphScheduleListItem({ schedule }: Args) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isViewOpen, setIsViewOpen] = useState(false);

  const { mutateAsync: deleteSchedule, isPending: isDeleting } =
    useDeleteV1DeleteExecutionSchedule();

  const nextRunDate = schedule.next_run_time
    ? new Date(schedule.next_run_time)
    : null;
  const nextRunLabel =
    nextRunDate && !Number.isNaN(nextRunDate.valueOf())
      ? `Next ${formatDistanceToNow(nextRunDate, { addSuffix: true })}`
      : "Pending";
  const nextRunTitle = nextRunDate ? nextRunDate.toString() : undefined;

  const recurrenceLabel = schedule.cron
    ? safeHumanizeCron(schedule.cron)
    : "Runs once";

  const agentLabel = schedule.agent_name || schedule.name || "Scheduled agent";
  const agentHref = `/build?flowID=${schedule.graph_id}&flowVersion=${schedule.graph_version}`;

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
      await deleteSchedule({ scheduleId: schedule.id });
      toast({ title: "Schedule deleted" });
      setIsDeleteOpen(false);
      invalidateAllScheduleQueries(queryClient, schedule.graph_id);
    } catch (error) {
      toast({
        title: "Failed to delete schedule",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  return {
    nextRunLabel,
    nextRunTitle,
    recurrenceLabel,
    agentLabel,
    agentHref,
    isDeleteOpen,
    openDelete,
    closeDelete,
    isDeleting,
    handleDelete,
    isViewOpen,
    openView,
    closeView,
  };
}

function safeHumanizeCron(cron: string): string {
  try {
    return humanizeCronExpression(cron);
  } catch {
    return cron;
  }
}
