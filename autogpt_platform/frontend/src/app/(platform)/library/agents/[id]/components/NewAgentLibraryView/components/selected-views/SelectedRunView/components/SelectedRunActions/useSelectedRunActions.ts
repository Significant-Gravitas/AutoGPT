"use client";

import {
  getGetV1ListGraphExecutionsInfiniteQueryOptions,
  usePostV1ExecuteGraphAgent,
  usePostV1StopGraphExecution,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Args {
  agentGraphId: string;
  run?: GraphExecution;
  onSelectRun?: (id: string) => void;
  onClearSelectedRun?: () => void;
}

export function useSelectedRunActions(args: Args) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const canStop =
    args.run?.status === "RUNNING" || args.run?.status === "QUEUED";

  const { mutateAsync: stopRun, isPending: isStopping } =
    usePostV1StopGraphExecution();

  const { mutateAsync: executeRun, isPending: isRunningAgain } =
    usePostV1ExecuteGraphAgent();

  async function handleStopRun() {
    try {
      await stopRun({
        graphId: args.run?.graph_id ?? "",
        graphExecId: args.run?.id ?? "",
      });

      toast({ title: "Run stopped" });

      await queryClient.invalidateQueries({
        queryKey: getGetV1ListGraphExecutionsInfiniteQueryOptions(
          args.agentGraphId,
        ).queryKey,
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
    if (!args.run) {
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
        graphId: args.run.graph_id,
        graphVersion: args.run.graph_version,
        data: {
          inputs: args.run.inputs || {},
          credentials_inputs: args.run.credential_inputs || {},
          source: "library",
        },
      });

      const newRunId = res?.status === 200 ? (res?.data?.id ?? "") : "";

      await queryClient.invalidateQueries({
        queryKey: getGetV1ListGraphExecutionsInfiniteQueryOptions(
          args.agentGraphId,
        ).queryKey,
      });

      if (newRunId && args.onSelectRun) args.onSelectRun(newRunId);
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
  const openInBuilderHref = args.run
    ? `/build?flowID=${args.run.graph_id}&flowVersion=${args.run.graph_version}&flowExecutionID=${args.run.id}`
    : undefined;

  return {
    openInBuilderHref,
    showDeleteDialog,
    canStop,
    isStopping,
    isRunningAgain,
    handleShowDeleteDialog,
    handleStopRun,
    handleRunAgain,
  } as const;
}
