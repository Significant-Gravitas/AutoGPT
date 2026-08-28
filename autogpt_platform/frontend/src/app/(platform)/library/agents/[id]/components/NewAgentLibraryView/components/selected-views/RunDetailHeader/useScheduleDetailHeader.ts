"use client";

import { useDeleteV1DeleteExecutionSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import { useQueryClient } from "@tanstack/react-query";
import { getBuilderHref } from "@/services/org-team/builder";

export function useScheduleDetailHeader(
  agentGraphId: string,
  scheduleId?: string,
  agentGraphVersion?: number | string,
  organizationId?: string | null,
  teamId?: string | null,
) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const deleteMutation = useDeleteV1DeleteExecutionSchedule({
    mutation: {
      onSuccess: () => {
        toast({ title: "Schedule deleted" });
        invalidateAllScheduleQueries(queryClient, agentGraphId);
      },
      onError: (error: any) =>
        toast({
          title: "Failed to delete schedule",
          description: error?.message || "An unexpected error occurred.",
          variant: "destructive",
        }),
    },
    request: getTenantRequestInit(organizationId, teamId),
  });

  function deleteSchedule() {
    if (!scheduleId) return;
    deleteMutation.mutate({ scheduleId });
  }

  const openInBuilderHref = getBuilderHref({
    graphId: agentGraphId,
    graphVersion: agentGraphVersion,
    organizationId: organizationId ?? null,
    teamId: teamId ?? null,
  });

  return {
    deleteSchedule,
    isDeleting: deleteMutation.isPending,
    openInBuilderHref,
  } as const;
}
