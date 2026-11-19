// Creating this hook, because we are using same saving stuff at multiple places in our builder

import { useCallback, useState } from "react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import {
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
import {
  describeSaveError,
  markInvalidFields,
} from "../helpers/saveGraphError";
import {
  getLoadedGraph,
  isSaveCancelled,
  waitForLoadedGraph,
} from "../helpers/waitForLoadedGraph";

export type SaveGraphOptions = {
  showToast?: boolean;
  onSuccess?: (graph: GraphModel) => void;
  onError?: (error: any) => void;
};

/** Needs useFlow mounted, which fills the loadedGraph that saves wait for. */
export const useSaveGraph = ({
  showToast = true,
  onSuccess,
  onError,
}: SaveGraphOptions) => {
  const { toast } = useToast();

  const [{ flowID, flowVersion }, setQueryStates] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
  });

  const setGraphSchemas = useGraphStore(
    useShallow((state) => state.setGraphSchemas),
  );

  // Saves waiting while the canvas is still loading the workflow they'd save.
  const [canvasWaits, setCanvasWaits] = useState(0);

  const { mutateAsync: createNewGraph, isPending: isCreating } =
    usePostV1CreateNewGraph({
      mutation: {
        onSuccess: async (response) => {
          const data = response.data as GraphModel;
          setQueryStates({
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
              title: "Workflow saved successfully",
              description: "The workflow has been saved successfully.",
              variant: "default",
            });
          }
        },
      },
    });

  const { mutateAsync: updateGraph, isPending: isUpdating } =
    usePutV1UpdateGraphVersion({
      mutation: {
        onSuccess: async (response, { graphId }) => {
          const data = (response.data as UpdateGraphResponse).graph;
          setQueryStates({
            flowID: data.id,
            flowVersion: data.version,
          });

          // A save can answer with a copy under a new ID; the draft sits under the one sent.
          for (const id of new Set([graphId, data.id])) {
            if (id) await draftService.deleteDraft(id);
          }

          onSuccess?.(data);
          if (showToast) {
            toast({
              title: "Workflow saved successfully",
              description: "The workflow has been saved successfully.",
              variant: "default",
            });
          }
        },
      },
    });

  const persistGraph = useCallback(
    async (values?: { name?: string; description?: string }) => {
      // A saved workflow is saved as a new version of the graph the canvas
      // shows, so while the canvas is still loading the one in the URL (just
      // after a save, or as the page opens) the save waits for it.
      let graph = flowID ? getLoadedGraph(flowID, flowVersion) : null;
      if (flowID && !graph) {
        setCanvasWaits((count) => count + 1);
        try {
          graph = await waitForLoadedGraph(flowID, flowVersion);
        } finally {
          setCanvasWaits((count) => count - 1);
        }
      }
      const graphNodes = useNodeStore.getState().getBackendNodes();
      const graphLinks = useEdgeStore.getState().getBackendLinks();

      if (graph && graph.id) {
        const data: Graph = {
          id: graph.id,
          name:
            values?.name ||
            graph.name ||
            `New Workflow ${new Date().toISOString()}`,
          description: values?.description ?? graph.description ?? "",
          nodes: graphNodes,
          links: graphLinks,
        };

        if (graphsEquivalent(graph, data)) {
          if (showToast) {
            toast({
              title: "No changes to save",
              description: "The workflow is the same as the saved version.",
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
          name: values?.name || `New Workflow ${new Date().toISOString()}`,
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
    [
      flowID,
      flowVersion,
      showToast,
      toast,
      createNewGraph,
      updateGraph,
      setGraphSchemas,
    ],
  );

  // Resolves to null when the save fails, and the user has been told why, or
  // when the builder left the workflow before it loaded.
  const saveGraph = useCallback(
    async (values?: { name?: string; description?: string }) => {
      let saved: GraphModel;
      try {
        saved = await persistGraph(values);
      } catch (error) {
        if (isSaveCancelled(error)) return null;
        onError?.(error);
        markInvalidFields(error);
        toast(describeSaveError(error));
        return null;
      }
      useNodeStore.getState().clearAllNodeErrors();
      return saved;
    },
    [persistGraph, onError, toast],
  );

  return {
    saveGraph,
    isSaving: canvasWaits > 0 || isCreating || isUpdating,
  };
};
