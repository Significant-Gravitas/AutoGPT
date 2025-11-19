// Creating this hook, because we are using same saving stuff at multiple places in our builder

import { useCallback } from "react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import {
  getGetV1GetSpecificGraphQueryKey,
  useGetV1GetSpecificGraph,
  usePostV1CreateNewGraph,
  usePutV1UpdateGraphVersion,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { Graph } from "@/app/api/__generated__/models/graph";
import { useNodeStore } from "../stores/nodeStore";
import { useEdgeStore } from "../stores/edgeStore";
import { graphsEquivalent } from "../components/NewControlPanel/NewSaveControl/helpers";

export type SaveGraphOptions = {
  showToast?: boolean;
  onSuccess?: (graph: GraphModel) => void;
  onError?: (error: any) => void;
};

export const useSaveGraph = ({
  showToast = true,
  onSuccess,
  onError,
}: SaveGraphOptions) => {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const [{ flowID, flowVersion }, setQueryStates] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
  });

  const { data: graph } = useGetV1GetSpecificGraph(
    flowID ?? "",
    flowVersion !== null ? { version: flowVersion } : {},
    {
      query: {
        select: (res) => res.data as GraphModel,
        enabled: !!flowID,
      },
    },
  );

  const { mutateAsync: createNewGraph, isPending: isCreating } =
    usePostV1CreateNewGraph({
      mutation: {
        onSuccess: (response) => {
          const data = response.data as GraphModel;
          setQueryStates({
            flowID: data.id,
            flowVersion: data.version,
          });
          queryClient.refetchQueries({
            queryKey: getGetV1GetSpecificGraphQueryKey(data.id),
          });
          onSuccess?.(data);
          if (showToast) {
            toast({
              title: "Graph saved successfully",
              description: "The graph has been saved successfully.",
              variant: "default",
            });
          }
        },
        onError: (error) => {
          onError?.(error);
          toast({
            title: "Error saving graph",
            description:
              (error as any).message ?? "An unexpected error occurred.",
            variant: "destructive",
          });
        },
      },
    });

  const { mutateAsync: updateGraph, isPending: isUpdating } =
    usePutV1UpdateGraphVersion({
      mutation: {
        onSuccess: (response) => {
          const data = response.data as GraphModel;
          setQueryStates({
            flowID: data.id,
            flowVersion: data.version,
          });
          queryClient.refetchQueries({
            queryKey: getGetV1GetSpecificGraphQueryKey(data.id),
          });
          onSuccess?.(data);
          if (showToast) {
            toast({
              title: "Graph saved successfully",
              description: "The graph has been saved successfully.",
              variant: "default",
            });
          }
        },
        onError: (error) => {
          onError?.(error);
          toast({
            title: "Error saving graph",
            description:
              (error as any).message ?? "An unexpected error occurred.",
            variant: "destructive",
          });
        },
      },
    });

  const saveGraph = useCallback(
    async (values?: { name?: string; description?: string }) => {
      const graphNodes = useNodeStore.getState().getBackendNodes();
      const graphLinks = useEdgeStore.getState().getBackendLinks();

      if (graph && graph.id) {
        const data: Graph = {
          id: graph.id,
          name:
            values?.name ||
            graph.name ||
            `New Agent ${new Date().toISOString()}`,
          description: values?.description ?? graph.description ?? "",
          nodes: graphNodes,
          links: graphLinks,
        };

        if (graphsEquivalent(graph, data)) {
          if (showToast) {
            toast({
              title: "No changes to save",
              description: "The graph is the same as the saved version.",
              variant: "default",
            });
          }
          return;
        }

        await updateGraph({ graphId: graph.id, data: data });
      } else {
        const data: Graph = {
          name: values?.name || `New Agent ${new Date().toISOString()}`,
          description: values?.description || "",
          nodes: graphNodes,
          links: graphLinks,
        };

        await createNewGraph({ data: { graph: data } });
      }
    },
    [graph, toast, createNewGraph, updateGraph],
  );

  return {
    saveGraph,
    isSaving: isCreating || isUpdating,
  };
};
