"use client";

import {
  getGetV1ListGraphExecutionsQueryKey,
  usePostV1ExecuteGraphAgent,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useDeleteV1DeleteExecutionSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { getBuilderHref } from "@/services/org-team/builder";

interface UseSelectedScheduleActionsProps {
  agent: LibraryAgent;
  scheduleId: string;
  schedule?: GraphExecutionJobInfo;
  onDeleted?: () => void;
  onSelectRun?: (id: string) => void;
}

export function useSelectedScheduleActions({
  agent,
  scheduleId,
  schedule,
  onDeleted,
  onSelectRun,
}: UseSelectedScheduleActionsProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const organizationId = schedule?.organization_id ?? agent.organization_id;
  const teamId = schedule?.team_id ?? agent.team_id;

  const deleteMutation = useDeleteV1DeleteExecutionSchedule({
    mutation: {
      onSuccess: () => {
        toast({ title: "Schedule deleted" });
        setShowDeleteDialog(false);

        onDeleted?.();

        invalidateAllScheduleQueries(queryClient, agent.graph_id);
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
    request: getTenantRequestInit(organizationId, teamId),
  });

  const { mutateAsync: executeAgent, isPending: isRunning } =
    usePostV1ExecuteGraphAgent({
      request: getTenantRequestInit(organizationId, teamId),
    });

  function handleDelete() {
    if (!scheduleId) return;
    deleteMutation.mutate({ scheduleId });
  }

  async function handleRunNow() {
    if (!schedule) {
      toast({
        title: "Schedule not loaded",
        description: "Please wait for the schedule to load.",
        variant: "destructive",
      });
      return;
    }

    try {
      toast({ title: "Run started" });

      const res = await executeAgent({
        graphId: schedule.graph_id,
        graphVersion: schedule.graph_version,
        data: {
          inputs: schedule.input_data || {},
          credentials_inputs: schedule.input_credentials || {},
          source: "library",
        },
      });

      const newRunID = okData(res)?.id;

      await queryClient.invalidateQueries({
        queryKey: getTeamScopedQueryKey(
          getGetV1ListGraphExecutionsQueryKey(agent.graph_id),
          organizationId,
          teamId,
        ),
      });

      if (newRunID && onSelectRun) {
        onSelectRun(newRunID);
      }
    } catch (error: unknown) {
      toast({
        title: "Failed to start run",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  const openInBuilderHref = getBuilderHref({
    graphId: agent.graph_id,
    graphVersion: agent.graph_version,
    organizationId: organizationId ?? null,
    teamId: teamId ?? null,
  });

  return {
    openInBuilderHref,
    showDeleteDialog,
    setShowDeleteDialog,
    handleDelete,
    isDeleting: deleteMutation.isPending,
    handleRunNow,
    isRunning,
  };
}
