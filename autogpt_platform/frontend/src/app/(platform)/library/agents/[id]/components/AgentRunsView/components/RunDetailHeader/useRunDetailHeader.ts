"use client";

import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import {
  usePostV1StopGraphExecution,
  getGetV1ListGraphExecutionsInfiniteQueryOptions,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useDeleteV1DeleteGraphExecution } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";

export function useRunDetailHeader(
  agentGraphId: string,
  run?: GraphExecution,
  onSelectRun?: (id: string) => void,
  onClearSelectedRun?: () => void,
) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const stopMutation = usePostV1StopGraphExecution({
    mutation: {
      onSuccess: () => {
        toast({ title: "Run stopped" });
        queryClient.invalidateQueries({
          queryKey:
            getGetV1ListGraphExecutionsInfiniteQueryOptions(agentGraphId)
              .queryKey,
        });
      },
      onError: (error: any) => {
        toast({
          title: "Failed to stop run",
          description: error?.message || "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  function stopRun() {
    if (!run) return;
    stopMutation.mutate({ graphId: run.graph_id, graphExecId: run.id });
  }

  const canStop = run?.status === "RUNNING" || run?.status === "QUEUED";

  // Delete run
  const deleteMutation = useDeleteV1DeleteGraphExecution({
    mutation: {
      onSuccess: () => {
        toast({ title: "Run deleted" });
        queryClient.invalidateQueries({
          queryKey:
            getGetV1ListGraphExecutionsInfiniteQueryOptions(agentGraphId)
              .queryKey,
        });
        if (onClearSelectedRun) onClearSelectedRun();
      },
      onError: (error: any) =>
        toast({
          title: "Failed to delete run",
          description: error?.message || "An unexpected error occurred.",
          variant: "destructive",
        }),
    },
  });

  function deleteRun() {
    if (!run) return;
    deleteMutation.mutate({ graphExecId: run.id });
  }

  // Run again (execute agent with previous inputs/credentials)
  const executeMutation = usePostV1ExecuteGraphAgent({
    mutation: {
      onSuccess: async (res) => {
        toast({ title: "Run started" });
        const newRunId = res?.status === 200 ? (res?.data?.id ?? "") : "";

        await queryClient.invalidateQueries({
          queryKey:
            getGetV1ListGraphExecutionsInfiniteQueryOptions(agentGraphId)
              .queryKey,
        });
        if (newRunId && onSelectRun) onSelectRun(newRunId);
      },
      onError: (error: any) =>
        toast({
          title: "Failed to start run",
          description: error?.message || "An unexpected error occurred.",
          variant: "destructive",
        }),
    },
  });

  function runAgain() {
    if (!run) return;
    executeMutation.mutate({
      graphId: run.graph_id,
      graphVersion: run.graph_version,
      data: {
        inputs: (run as any).inputs || {},
        credentials_inputs: (run as any).credential_inputs || {},
      },
    } as any);
  }

  // Open in builder URL helper
  const openInBuilderHref = run
    ? `/build?flowID=${run.graph_id}&flowVersion=${run.graph_version}&flowExecutionID=${run.id}`
    : undefined;

  return {
    stopRun,
    canStop,
    isStopping: stopMutation.isPending,
    deleteRun,
    isDeleting: deleteMutation.isPending,
    runAgain,
    isRunningAgain: executeMutation.isPending,
    openInBuilderHref,
  } as const;
}
