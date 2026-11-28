"use client";

import {
  getGetV1ListGraphExecutionsQueryKey,
  usePostV1ExecuteGraphAgent,
  usePostV1StopGraphExecution,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV2ListPresetsQueryKey,
  usePostV2CreateANewPreset,
} from "@/app/api/__generated__/endpoints/presets/presets";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  CredentialsActionsContext,
  CredentialsProvidersContext,
} from "@/providers/agent-credentials/credentials-provider";
import { useQueryClient } from "@tanstack/react-query";
import { useContext, useState } from "react";
import { formatValidationError } from "../../../../modals/RunAgentModal/errorHelpers";
import {
  describeDeletedKeys,
  getDeletedKeys,
  getRefusedKeys,
  getRerunForm,
  RerunForm,
} from "./helpers";
import { useRunVersionAgent } from "./useRunVersionAgent";

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
  const allProviders = useContext(CredentialsProvidersContext);
  const credentialsActions = useContext(CredentialsActionsContext);

  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [isCreateTemplateModalOpen, setIsCreateTemplateModalOpen] =
    useState(false);
  // The run form a rerun opens to replace the run's deleted keys
  const [rerunForm, setRerunForm] = useState<RerunForm | null>(null);
  // It reruns the workflow version the run ran, read when it isn't the current one
  const runVersion = useRunVersionAgent({
    agent,
    run,
    enabled: rerunForm !== null,
  });

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
        queryKey: getGetV1ListGraphExecutionsQueryKey(agentGraphId),
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

    // A key the run used was deleted: the server would refuse it, so the run
    // form opens with the run's inputs for the user to pick another key
    const deletedKeys = getDeletedKeys(run.credential_inputs, allProviders);
    if (deletedKeys.length > 0) {
      setRerunForm(getRerunForm(run, deletedKeys));
      return;
    }

    try {
      const res = await executeRun({
        graphId: run.graph_id,
        graphVersion: run.graph_version,
        data: {
          inputs: run.inputs || {},
          credentials_inputs: run.credential_inputs || {},
          source: "library",
        },
      });
      toast({ title: "Run started" });

      const newRunId = okData(res)?.id;

      await queryClient.invalidateQueries({
        queryKey: getGetV1ListGraphExecutionsQueryKey(agentGraphId),
      });

      if (newRunId && onSelectRun) onSelectRun(newRunId);
    } catch (error: unknown) {
      handleRunAgainError(run, error);
    }
  }

  function handleRunAgainError(failedRun: GraphExecution, error: unknown) {
    // Deleted since the page read the user's keys, e.g. in another tab: they
    // are read again, and the run form opens to pick another
    const refusedKeys = getRefusedKeys(
      error,
      failedRun.credential_inputs,
      allProviders,
    );
    if (refusedKeys.length > 0) {
      credentialsActions?.reload();
      setRerunForm(getRerunForm(failedRun, refusedKeys));
      toast({
        title: "Failed to start run",
        description: describeDeletedKeys(refusedKeys),
        variant: "destructive",
      });
      return;
    }
    // Each step's problem, not only how many there are
    toast({
      title: "Failed to start run",
      description:
        error instanceof Error
          ? formatValidationError(error, { graph_id: failedRun.graph_id })
          : "An unexpected error occurred.",
      variant: "destructive",
      duration: 10000,
    });
  }

  function handleRerunFormOpenChange(open: boolean) {
    if (!open) setRerunForm(null);
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
      return false;
    }

    try {
      // A failed save throws: the API client raises on any non-2xx answer.
      await createPreset({
        data: {
          name,
          description,
          graph_execution_id: run.id,
        },
      });

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

      return true;
    } catch (error: unknown) {
      toast({
        title: "Failed to create template",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
      return false;
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
    rerunForm,
    // The workflow at the run's version, once it's known
    rerunAgent: rerunForm ? runVersion.runAgent : undefined,
    isOpeningRerunForm: rerunForm !== null && runVersion.isLoading,
    hasRerunVersionError: rerunForm !== null && runVersion.hasError,
    isRetryingRerunVersion: runVersion.isRetrying,
    retryRerunVersion: runVersion.retry,
    handleRerunFormOpenChange,
    handleCreateTemplate,
    isCreatingTemplate,
    isCreateTemplateModalOpen,
    setIsCreateTemplateModalOpen,
  } as const;
}
