import {
  usePostV1ExecuteGraphAgent,
  usePostV1StopGraphExecution,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useNewSaveControl } from "../../../NewControlPanel/NewSaveControl/useNewSaveControl";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { GraphExecutionMeta } from "@/app/(platform)/library/agents/[id]/components/OldAgentLibraryView/use-agent-runs";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useShallow } from "zustand/react/shallow";
import { useState } from "react";

export const useRunGraph = () => {
  const { onSubmit: onSaveGraph, isLoading: isSaving } = useNewSaveControl({
    showToast: false,
  });
  const { toast } = useToast();
  const setIsGraphRunning = useGraphStore(
    useShallow((state) => state.setIsGraphRunning),
  );
  const hasInputs = useGraphStore(useShallow((state) => state.hasInputs));
  const hasCredentials = useGraphStore(
    useShallow((state) => state.hasCredentials),
  );
  const [openRunInputDialog, setOpenRunInputDialog] = useState(false);

  const [{ flowID, flowVersion, flowExecutionID }, setQueryStates] =
    useQueryStates({
      flowID: parseAsString,
      flowVersion: parseAsInteger,
      flowExecutionID: parseAsString,
    });

  const { mutateAsync: executeGraph } = usePostV1ExecuteGraphAgent({
    mutation: {
      onSuccess: (response) => {
        const { id } = response.data as GraphExecutionMeta;
        setQueryStates({
          flowExecutionID: id,
        });
      },
      onError: (error) => {
        setIsGraphRunning(false);

        toast({
          title: (error.detail as string) ?? "An unexpected error occurred.",
          description: "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  const { mutateAsync: stopGraph } = usePostV1StopGraphExecution({
    mutation: {
      onSuccess: () => {
        setIsGraphRunning(false);
      },
      onError: (error) => {
        toast({
          title: (error.detail as string) ?? "An unexpected error occurred.",
          description: "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  const handleRunGraph = async () => {
    await onSaveGraph(undefined);

    if (hasInputs() || hasCredentials()) {
      setOpenRunInputDialog(true);
    } else {
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
    openRunInputDialog,
    setOpenRunInputDialog,
  };
};
