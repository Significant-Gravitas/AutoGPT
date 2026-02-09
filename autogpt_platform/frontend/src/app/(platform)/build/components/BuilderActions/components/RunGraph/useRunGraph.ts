import {
  usePostV1ExecuteGraphAgent,
  usePostV1StopGraphExecution,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { GraphExecutionMeta } from "@/app/(platform)/library/agents/[id]/components/OldAgentLibraryView/use-agent-runs";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useShallow } from "zustand/react/shallow";
import { useEffect, useState } from "react";
import { useSaveGraph } from "@/app/(platform)/build/hooks/useSaveGraph";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { ApiError } from "@/lib/autogpt-server-api/helpers"; // Check if this exists
import { useTutorialStore } from "@/app/(platform)/build/stores/tutorialStore";

export const useRunGraph = () => {
  const { saveGraph, isSaving } = useSaveGraph({
    showToast: false,
  });
  const { toast } = useToast();
  const hasInputs = useGraphStore(useShallow((state) => state.hasInputs));
  const hasCredentials = useGraphStore(
    useShallow((state) => state.hasCredentials),
  );
  const setIsGraphRunning = useGraphStore(
    useShallow((state) => state.setIsGraphRunning),
  );
  const [openRunInputDialog, setOpenRunInputDialog] = useState(false);

  const setNodeErrorsForBackendId = useNodeStore(
    useShallow((state) => state.setNodeErrorsForBackendId),
  );
  const clearAllNodeErrors = useNodeStore(
    useShallow((state) => state.clearAllNodeErrors),
  );

  // Tutorial integration - force open dialog when tutorial requests it
  const forceOpenRunInputDialog = useTutorialStore(
    (state) => state.forceOpenRunInputDialog,
  );
  const setForceOpenRunInputDialog = useTutorialStore(
    (state) => state.setForceOpenRunInputDialog,
  );

  // Sync tutorial state with dialog state
  useEffect(() => {
    if (forceOpenRunInputDialog && !openRunInputDialog) {
      setOpenRunInputDialog(true);
    }
  }, [forceOpenRunInputDialog, openRunInputDialog]);

  // Reset tutorial state when dialog closes
  const handleSetOpenRunInputDialog = (isOpen: boolean) => {
    setOpenRunInputDialog(isOpen);
    if (!isOpen && forceOpenRunInputDialog) {
      setForceOpenRunInputDialog(false);
    }
  };

  const [{ flowID, flowVersion, flowExecutionID }, setQueryStates] =
    useQueryStates({
      flowID: parseAsString,
      flowVersion: parseAsInteger,
      flowExecutionID: parseAsString,
    });

  const { mutateAsync: executeGraph, isPending: isExecutingGraph } =
    usePostV1ExecuteGraphAgent({
      mutation: {
        onSuccess: (response: any) => {
          clearAllNodeErrors();
          const { id } = response.data as GraphExecutionMeta;
          setQueryStates({
            flowExecutionID: id,
          });
        },
        onError: (error: any) => {
          setIsGraphRunning(false);
          if (error instanceof ApiError && error.isGraphValidationError?.()) {
            const errorData = error.response?.detail;

            if (errorData?.node_errors) {
              Object.entries(errorData.node_errors).forEach(
                ([backendId, nodeErrors]) => {
                  setNodeErrorsForBackendId(
                    backendId,
                    nodeErrors as { [key: string]: string },
                  );
                },
              );

              useNodeStore.getState().nodes.forEach((node) => {
                const backendId = node.data.metadata?.backend_id || node.id;
                if (!errorData.node_errors[backendId as string]) {
                  useNodeStore.getState().updateNodeErrors(node.id, {});
                }
              });
            }

            toast({
              title: errorData?.message || "Graph validation failed",
              description:
                "Please fix the validation errors on the highlighted nodes and try again.",
              variant: "destructive",
            });
          } else {
            toast({
              title:
                (error.detail as string) ?? "An unexpected error occurred.",
              description: "An unexpected error occurred.",
              variant: "destructive",
            });
          }
        },
      },
    });

  const { mutateAsync: stopGraph, isPending: isTerminatingGraph } =
    usePostV1StopGraphExecution({
      mutation: {
        onSuccess: () => {},
        onError: (error: any) => {
          toast({
            title: (error.detail as string) ?? "An unexpected error occurred.",
            description: "An unexpected error occurred.",
            variant: "destructive",
          });
        },
      },
    });

  const handleRunGraph = async () => {
    await saveGraph(undefined);

    if (hasInputs() || hasCredentials()) {
      setOpenRunInputDialog(true);
    } else {
      // Optimistically set running state immediately for responsive UI
      setIsGraphRunning(true);
      await executeGraph({
        graphId: flowID ?? "",
        graphVersion: flowVersion || null,
        data: { inputs: {}, credentials_inputs: {}, source: "builder" },
      });
    }
  };

  const handleStopGraph = async () => {
    if (!flowExecutionID) {
      return;
    }
    await stopGraph({
      graphId: flowID ?? "",
      graphExecId: flowExecutionID,
    });
  };

  return {
    handleRunGraph,
    handleStopGraph,
    isSaving,
    isExecutingGraph,
    isTerminatingGraph,
    openRunInputDialog,
    setOpenRunInputDialog: handleSetOpenRunInputDialog,
  };
};
