import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { GraphExecutionMeta } from "@/lib/autogpt-server-api";
import { parseAsString, useQueryStates } from "nuqs";
import { useShallow } from "zustand/react/shallow";

export const useRunInputDialog = () => {
  const [{}, setQueryStates] = useQueryStates({
    flowExecutionID: parseAsString,
  });
  const setIsGraphRunning = useGraphStore(
    useShallow((state) => state.setIsGraphRunning),
  );
  const { toast } = useToast();
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
  return {
    executeGraph,
  };
};
