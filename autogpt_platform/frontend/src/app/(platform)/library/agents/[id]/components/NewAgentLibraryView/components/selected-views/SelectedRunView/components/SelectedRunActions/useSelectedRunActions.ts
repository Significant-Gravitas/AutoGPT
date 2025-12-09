"use client";

import {
  getGetV1ListGraphExecutionsInfiniteQueryOptions,
  usePostV1ExecuteGraphAgent,
  usePostV1StopGraphExecution,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV2ListPresetsQueryKey,
  usePostV2CreateANewPreset,
} from "@/app/api/__generated__/endpoints/presets/presets";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Args {
  agentGraphId: string;
  run?: GraphExecution;
  agent?: LibraryAgent;
  onSelectRun?: (id: string) => void;
  onClearSelectedRun?: () => void;
}

export function useSelectedRunActions(args: Args) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [isCreateTemplateModalOpen, setIsCreateTemplateModalOpen] =
    useState(false);

  const canStop =
    args.run?.status === "RUNNING" || args.run?.status === "QUEUED";

  const { mutateAsync: stopRun, isPending: isStopping } =
    usePostV1StopGraphExecution();

  const { mutateAsync: executeRun, isPending: isRunningAgain } =
    usePostV1ExecuteGraphAgent();

  const { mutateAsync: createPreset, isPending: isCreatingTemplate } =
    usePostV2CreateANewPreset();

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

  async function handleCreateTemplate(name: string, description: string) {
    if (!args.run) {
      toast({
        title: "Run not found",
        description: "Cannot create template from missing run",
        variant: "destructive",
      });
      return;
    }

    try {
      const res = await createPreset({
        data: {
          name,
          description,
          graph_execution_id: args.run.id,
        },
      });

      if (res.status === 200) {
        toast({
          title: "Template created",
        });

        if (args.agent) {
          queryClient.invalidateQueries({
            queryKey: getGetV2ListPresetsQueryKey({
              graph_id: args.agent.graph_id,
            }),
          });
        }

        setIsCreateTemplateModalOpen(false);
      }
    } catch (error: unknown) {
      toast({
        title: "Failed to create template",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
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
    handleCreateTemplate,
    isCreatingTemplate,
    isCreateTemplateModalOpen,
    setIsCreateTemplateModalOpen,
  } as const;
}
