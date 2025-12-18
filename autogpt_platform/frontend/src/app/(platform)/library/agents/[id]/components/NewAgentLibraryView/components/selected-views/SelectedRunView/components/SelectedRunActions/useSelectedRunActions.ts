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

interface Params {
  agentGraphId: string;
  run?: GraphExecution;
  agent?: LibraryAgent;
  onSelectRun?: (id: string) => void;
}

export function useSelectedRunActions({
  agentGraphId,
  run,
  agent,
  onSelectRun,
}: Params) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [isCreateTemplateModalOpen, setIsCreateTemplateModalOpen] =
    useState(false);

  const canStop = run?.status === "RUNNING" || run?.status === "QUEUED";

  const canRunManually = !agent?.trigger_setup_info;

  const { mutateAsync: stopRun, isPending: isStopping } =
    usePostV1StopGraphExecution();

  const { mutateAsync: executeRun, isPending: isRunningAgain } =
    usePostV1ExecuteGraphAgent();

  const { mutateAsync: createPreset, isPending: isCreatingTemplate } =
    usePostV2CreateANewPreset();

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
          inputs: run.inputs || {},
          credentials_inputs: run.credential_inputs || {},
          source: "library",
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

  async function handleCreateTemplate(name: string, description: string) {
    if (!run) {
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
          graph_execution_id: run.id,
        },
      });

      if (res.status === 200) {
        toast({
          title: "Template created",
        });

        if (agent) {
          queryClient.invalidateQueries({
            queryKey: getGetV2ListPresetsQueryKey({
              graph_id: agent.graph_id,
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
  const openInBuilderHref = run
    ? `/build?flowID=${run.graph_id}&flowVersion=${run.graph_version}&flowExecutionID=${run.id}`
    : undefined;

  return {
    openInBuilderHref,
    showDeleteDialog,
    canStop,
    isStopping,
    canRunManually,
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
