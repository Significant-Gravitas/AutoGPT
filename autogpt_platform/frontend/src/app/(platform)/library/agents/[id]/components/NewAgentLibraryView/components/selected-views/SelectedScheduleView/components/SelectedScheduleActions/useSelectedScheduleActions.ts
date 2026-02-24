"use client";

import {
  getGetV1ListExecutionSchedulesForAGraphQueryOptions,
  useDeleteV1DeleteExecutionSchedule,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface UseSelectedScheduleActionsProps {
  agent: LibraryAgent;
  scheduleId: string;
  onDeleted?: () => void;
}

export function useSelectedScheduleActions({
  agent,
  scheduleId,
  onDeleted,
}: UseSelectedScheduleActionsProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const deleteMutation = useDeleteV1DeleteExecutionSchedule({
    mutation: {
      onSuccess: () => {
        toast({ title: "Schedule deleted" });
        queryClient.invalidateQueries({
          queryKey: getGetV1ListExecutionSchedulesForAGraphQueryOptions(
            agent.graph_id,
          ).queryKey,
        });
        setShowDeleteDialog(false);
        onDeleted?.();
      },
      onError: (error: unknown) =>
        toast({
          title: "Failed to delete schedule",
          description:
            error instanceof Error
              ? error.message
              : "An unexpected error occurred.",
          variant: "destructive",
        }),
    },
  });

  function handleDelete() {
    if (!scheduleId) return;
    deleteMutation.mutate({ scheduleId });
  }

  const openInBuilderHref = `/build?flowID=${agent.graph_id}&flowVersion=${agent.graph_version}`;

  return {
    openInBuilderHref,
    showDeleteDialog,
    setShowDeleteDialog,
    handleDelete,
    isDeleting: deleteMutation.isPending,
  };
}
