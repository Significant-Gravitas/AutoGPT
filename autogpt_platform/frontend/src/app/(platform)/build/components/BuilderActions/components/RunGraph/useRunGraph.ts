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
          const { id } = response.data as GraphExecutionMeta;
          setQueryStates({
            flowExecutionID: id,
          });
        },
        onError: (error: any) => {
          // Reset running state on error
          setIsGraphRunning(false);
          toast({
            title: (error.detail as string) ?? "An unexpected error occurred.",
            description: "An unexpected error occurred.",
            variant: "destructive",
          });
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
        data: { inputs: {}, credentials_inputs: {} },
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
