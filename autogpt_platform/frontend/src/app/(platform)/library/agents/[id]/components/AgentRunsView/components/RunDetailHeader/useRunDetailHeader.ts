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
import { useState } from "react";

export function useRunDetailHeader(
  agentGraphId: string,
  run?: GraphExecution,
  onSelectRun?: (id: string) => void,
  onClearSelectedRun?: () => void,
) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const canStop = run?.status === "RUNNING" || run?.status === "QUEUED";

  const { mutateAsync: stopRun, isPending: isStopping } =
    usePostV1StopGraphExecution();

  const { mutateAsync: deleteRun, isPending: isDeleting } =
    useDeleteV1DeleteGraphExecution();

  const { mutateAsync: executeRun, isPending: isRunningAgain } =
    usePostV1ExecuteGraphAgent();

  async function handleDeleteRun() {
    try {
      await deleteRun({ graphExecId: run?.id ?? "" });

      toast({ title: "Run deleted" });

      await queryClient.refetchQueries({
        queryKey:
          getGetV1ListGraphExecutionsInfiniteQueryOptions(agentGraphId)
            .queryKey,
      });

      if (onClearSelectedRun) onClearSelectedRun();

      setShowDeleteDialog(false);
    } catch (error: unknown) {
      toast({
        title: "Failed to delete run",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  async function handleStopRun() {
    try {
      await stopRun({
        graphId: run?.graph_id ?? "",
        graphExecId: run?.id ?? "",
      });

      toast({ title: "Run stopped" });

      await queryClient.invalidateQueries({
        queryKey:
          getGetV1ListGraphExecutionsInfiniteQueryOptions(agentGraphId)
            .queryKey,
      });
    } catch (error: unknown) {
      toast({
        title: "Failed to stop run",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  async function handleRunAgain() {
    if (!run) {
      toast({
        title: "Run not found",
        description: "Run not found",
        variant: "destructive",
      });
      return;
    }

    try {
      toast({ title: "Run started" });

      const res = await executeRun({
        graphId: run.graph_id,
        graphVersion: run.graph_version,
        data: {
          inputs: (run as any).inputs || {},
          credentials_inputs: (run as any).credential_inputs || {},
        },
      });

      const newRunId = res?.status === 200 ? (res?.data?.id ?? "") : "";

      await queryClient.invalidateQueries({
        queryKey:
          getGetV1ListGraphExecutionsInfiniteQueryOptions(agentGraphId)
            .queryKey,
      });

      if (newRunId && onSelectRun) onSelectRun(newRunId);
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

  function handleShowDeleteDialog(open: boolean) {
    setShowDeleteDialog(open);
  }

  // Open in builder URL helper
  const openInBuilderHref = run
    ? `/build?flowID=${run.graph_id}&flowVersion=${run.graph_version}&flowExecutionID=${run.id}`
    : undefined;

  return {
    openInBuilderHref,
    showDeleteDialog,
    canStop,
    isStopping,
    isDeleting,
    isRunning: run?.status === "RUNNING",
    isRunningAgain,
    handleShowDeleteDialog,
    handleDeleteRun,
    handleStopRun,
    handleRunAgain,
  } as const;
}
