/**
 * REL-004 classification — CANONICAL mutable owner for flowID/flowVersion:
 * - Fields read: flowID, flowVersion
 * - Calls setter: YES — setBuilderQueryStates({flowID, flowVersion}) on
 *   create/update success (lines 64-66, 101-104) — single authority for
 *   advancing URL version
 * - Hydrates/mutates graph state indirectly: YES — setGraphSchemas,
 *   draftService.deleteDraft, graphsEquivalent baseline check
 * - Canonical mutable owner: useBuilderQueryStates
 *   (frontend/src/app/(platform)/build/hooks/useBuilderQueryStates.ts:18)
 *   — this hook now delegates URL writes to that hook; failed save leaves
 *   local edits, baseline, hash, version untouched (onError does not advance)
 * Verdict: canonical writer — migrated to useBuilderQueryStates
 */
// Creating this hook, because we are using same saving stuff at multiple places in our builder

import { useCallback } from "react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useBuilderQueryStates } from "./useBuilderQueryStates";
import {
  useGetV1GetSpecificGraph,
  usePostV1CreateNewGraph,
  usePutV1UpdateGraphVersion,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { Graph } from "@/app/api/__generated__/models/graph";
import { UpdateGraphResponse } from "@/app/api/__generated__/models/updateGraphResponse";
import { notifySkippedWebhookPresets } from "../helpers/skippedWebhookPresets";
import { useNodeStore } from "../stores/nodeStore";
import { useEdgeStore } from "../stores/edgeStore";
import { graphsEquivalent } from "../components/NewControlPanel/NewSaveControl/helpers";
import { useGraphStore } from "../stores/graphStore";
import { useShallow } from "zustand/react/shallow";
import {
  draftService,
  clearTempFlowId,
  getTempFlowId,
} from "@/services/builder-draft/draft-service";

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

  const [{ flowID, flowVersion }, setBuilderQueryStates] =
    useBuilderQueryStates();

  const setGraphSchemas = useGraphStore(
    useShallow((state) => state.setGraphSchemas),
  );

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
        onSuccess: async (response) => {
          const data = response.data as GraphModel;
          setBuilderQueryStates({
            flowID: data.id,
            flowVersion: data.version,
          });

          const tempFlowId = getTempFlowId();
          if (tempFlowId) {
            await draftService.deleteDraft(tempFlowId);
            clearTempFlowId();
          }

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
          // REL-004: failed save must preserve local edits and not advance
          // saved baseline/hash/version — do not touch URL, draft, or schemas
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
        onSuccess: async (response) => {
          const data = (response.data as UpdateGraphResponse).graph;
          setBuilderQueryStates({
            flowID: data.id,
            flowVersion: data.version,
          });

          // Clear the draft for this flow after successful save
          if (data.id) {
            await draftService.deleteDraft(data.id);
          }

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
          // REL-004: failed save must not advance baseline/hash/version
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
          return graph;
        }

        const response = await updateGraph({ graphId: graph.id, data: data });
        const result = response.data as UpdateGraphResponse;
        const graphData = result.graph;
        setGraphSchemas(
          graphData.input_schema,
          graphData.credentials_input_schema,
          graphData.output_schema,
        );
        notifySkippedWebhookPresets(toast, result.skipped_webhook_presets);
        return graphData;
      } else {
        const data: Graph = {
          name: values?.name || `New Agent ${new Date().toISOString()}`,
          description: values?.description || "",
          nodes: graphNodes,
          links: graphLinks,
        };

        const response = await createNewGraph({
          data: { graph: data, source: "builder" },
        });
        const graphData = response.data as GraphModel;
        setGraphSchemas(
          graphData.input_schema,
          graphData.credentials_input_schema,
          graphData.output_schema,
        );
        return graphData;
      }
    },
    [graph, toast, createNewGraph, updateGraph],
  );

  return {
    saveGraph,
    isSaving: isCreating || isUpdating,
  };
};
