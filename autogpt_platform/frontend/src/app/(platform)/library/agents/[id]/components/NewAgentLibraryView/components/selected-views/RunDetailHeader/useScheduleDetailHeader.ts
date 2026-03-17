"use client";

import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import {
  useDeleteV1DeleteExecutionSchedule,
  getGetV1ListExecutionSchedulesForAGraphQueryOptions,
} from "@/app/api/__generated__/endpoints/schedules/schedules";

export function useScheduleDetailHeader(
  agentGraphId: string,
  scheduleId?: string,
  agentGraphVersion?: number | string,
) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const deleteMutation = useDeleteV1DeleteExecutionSchedule({
    mutation: {
      onSuccess: () => {
        toast({ title: "Schedule deleted" });
        queryClient.invalidateQueries({
          queryKey:
            getGetV1ListExecutionSchedulesForAGraphQueryOptions(agentGraphId)
              .queryKey,
        });
      },
      onError: (error: any) =>
        toast({
          title: "Failed to delete schedule",
          description: error?.message || "An unexpected error occurred.",
          variant: "destructive",
        }),
    },
  });

  function deleteSchedule() {
    if (!scheduleId) return;
    deleteMutation.mutate({ scheduleId });
  }

  const openInBuilderHref = `/build?flowID=${agentGraphId}&flowVersion=${agentGraphVersion}`;

  return {
    deleteSchedule,
    isDeleting: deleteMutation.isPending,
    openInBuilderHref,
  } as const;
}
