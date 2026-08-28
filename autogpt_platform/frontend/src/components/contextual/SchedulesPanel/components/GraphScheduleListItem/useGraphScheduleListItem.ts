import { useDeleteV1DeleteExecutionSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import { useQueryClient } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import { useState } from "react";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";
import { getBuilderHref } from "@/services/org-team/builder";

interface Args {
  schedule: GraphExecutionJobInfo;
}

export function useGraphScheduleListItem({ schedule }: Args) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isViewOpen, setIsViewOpen] = useState(false);

  const { mutateAsync: deleteSchedule, isPending: isDeleting } =
    useDeleteV1DeleteExecutionSchedule({
      request: getTenantRequestInit(schedule.organization_id, schedule.team_id),
    });

  const nextRunDate = schedule.next_run_time
    ? new Date(schedule.next_run_time)
    : null;
  const nextRunRelative =
    nextRunDate && !Number.isNaN(nextRunDate.valueOf())
      ? formatDistanceToNow(nextRunDate, { addSuffix: true })
      : null;
  const nextRunLabel = nextRunRelative ? `Next ${nextRunRelative}` : "Pending";
  const nextRunTitle = nextRunDate ? nextRunDate.toString() : undefined;

  const recurrenceLabel = schedule.cron
    ? safeHumanizeCronExpression(schedule.cron)
    : "Runs once";

  const agentLabel = schedule.agent_name || schedule.name || "Scheduled agent";
  const agentHref = getBuilderHref({
    graphId: schedule.graph_id,
    graphVersion: schedule.graph_version,
    organizationId: schedule.organization_id ?? null,
    teamId: schedule.team_id ?? null,
  });

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
    nextRunRelative,
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
