import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useNewSaveControl } from "../../../NewControlPanel/NewSaveControl/useNewSaveControl";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { GraphExecutionMeta } from "@/app/(platform)/library/agents/[id]/components/OldAgentLibraryView/use-agent-runs";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useShallow } from "zustand/react/shallow";

export const useRunGraph = () => {
  const { onSubmit: onSaveGraph, isLoading: isSaving } = useNewSaveControl({
    showToast: false,
  });
  const { toast } = useToast();
  const setIsGraphRunning = useGraphStore(
    useShallow((state) => state.setIsGraphRunning),
  );
  const [{ flowID, flowVersion }, setQueryStates] = useQueryStates({
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

  const runGraph = async () => {
    setIsGraphRunning(true);
    await onSaveGraph(undefined);

    // Todo : We need to save graph which has inputs and credentials inputs
    await executeGraph({
      graphId: flowID ?? "",
      graphVersion: flowVersion || null,
      data: {
        inputs: {},
        credentials_inputs: {},
      },
    });
  };

  return {
    runGraph,
    isSaving,
  };
};
