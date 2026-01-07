import { useState, useCallback, useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import {
  useNodeStore,
  NodeResolutionData,
} from "@/app/(platform)/build/stores/nodeStore";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import {
  useSubAgentUpdate,
  createUpdatedAgentNodeInputs,
  getBrokenEdgeIDs,
} from "@/app/(platform)/build/hooks/useSubAgentUpdate";
import { GraphInputSchema, GraphOutputSchema } from "@/lib/autogpt-server-api";
import { CustomNodeData } from "../../CustomNode";

// Stable empty set to avoid creating new references in selectors
const EMPTY_SET: Set<string> = new Set();

type UseSubAgentUpdateParams = {
  nodeID: string;
  nodeData: CustomNodeData;
};

export function useSubAgentUpdateState({
  nodeID,
  nodeData,
}: UseSubAgentUpdateParams) {
  const [showIncompatibilityDialog, setShowIncompatibilityDialog] =
    useState(false);

  // Get store actions
  const updateNodeData = useNodeStore(
    useShallow((state) => state.updateNodeData),
  );
  const setNodeResolutionMode = useNodeStore(
    useShallow((state) => state.setNodeResolutionMode),
  );
  const isNodeInResolutionMode = useNodeStore(
    useShallow((state) => state.isNodeInResolutionMode),
  );
  const setBrokenEdgeIDs = useNodeStore(
    useShallow((state) => state.setBrokenEdgeIDs),
  );
  // Get this node's broken edge IDs from the per-node map
  // Use EMPTY_SET as fallback to maintain referential stability
  const brokenEdgeIDs = useNodeStore(
    (state) => state.brokenEdgeIDs.get(nodeID) || EMPTY_SET,
  );
  const getNodeResolutionData = useNodeStore(
    useShallow((state) => state.getNodeResolutionData),
  );
  const connectedEdges = useEdgeStore(
    useShallow((state) => state.getNodeEdges(nodeID)),
  );
  const availableSubGraphs = useGraphStore(
    useShallow((state) => state.availableSubGraphs),
  );

  // Extract agent-specific data
  const graphID = nodeData.hardcodedValues?.graph_id as string | undefined;
  const graphVersion = nodeData.hardcodedValues?.graph_version as
    | number
    | undefined;
  const currentInputSchema = nodeData.hardcodedValues?.input_schema as
    | GraphInputSchema
    | undefined;
  const currentOutputSchema = nodeData.hardcodedValues?.output_schema as
    | GraphOutputSchema
    | undefined;

  // Use the sub-agent update hook
  const updateInfo = useSubAgentUpdate(
    nodeID,
    graphID,
    graphVersion,
    currentInputSchema,
    currentOutputSchema,
    connectedEdges,
    availableSubGraphs,
  );

  const isInResolutionMode = isNodeInResolutionMode(nodeID);

  // Handle update button click
  const handleUpdateClick = useCallback(() => {
    if (!updateInfo.hasUpdate || !updateInfo.latestGraph) return;

    if (updateInfo.isCompatible) {
      // Compatible update - apply directly
      const newHardcodedValues = createUpdatedAgentNodeInputs(
        nodeData.hardcodedValues,
        updateInfo.latestGraph,
      );
      updateNodeData(nodeID, { hardcodedValues: newHardcodedValues });
    } else {
      // Incompatible update - show dialog
      setShowIncompatibilityDialog(true);
    }
  }, [
    updateInfo.hasUpdate,
    updateInfo.latestGraph,
    updateInfo.isCompatible,
    nodeData.hardcodedValues,
    updateNodeData,
    nodeID,
  ]);

  // Handle confirming an incompatible update
  function handleConfirmIncompatibleUpdate() {
    if (!updateInfo.latestGraph || !updateInfo.incompatibilities) return;

    const latestGraph = updateInfo.latestGraph;

    // Get the new schemas from the latest graph version
    const newInputSchema =
      (latestGraph.input_schema as Record<string, unknown>) || {};
    const newOutputSchema =
      (latestGraph.output_schema as Record<string, unknown>) || {};

    // Create the updated hardcoded values but DON'T apply them yet
    // We'll apply them when resolution is complete
    const pendingHardcodedValues = createUpdatedAgentNodeInputs(
      nodeData.hardcodedValues,
      latestGraph,
    );

    // Get broken edge IDs and store them for this node
    const brokenIds = getBrokenEdgeIDs(
      connectedEdges,
      updateInfo.incompatibilities,
      nodeID,
    );
    setBrokenEdgeIDs(nodeID, brokenIds);

    // Enter resolution mode with both old and new schemas
    // DON'T apply the update yet - keep old schema so connections remain visible
    const resolutionData: NodeResolutionData = {
      incompatibilities: updateInfo.incompatibilities,
      pendingUpdate: {
        input_schema: newInputSchema,
        output_schema: newOutputSchema,
      },
      currentSchema: {
        input_schema: (currentInputSchema as Record<string, unknown>) || {},
        output_schema: (currentOutputSchema as Record<string, unknown>) || {},
      },
      pendingHardcodedValues,
    };
    setNodeResolutionMode(nodeID, true, resolutionData);

    setShowIncompatibilityDialog(false);
  }

  // Check if resolution is complete (all broken edges removed)
  const resolutionData = getNodeResolutionData(nodeID);

  // Auto-check resolution on edge changes
  useEffect(() => {
    if (!isInResolutionMode) return;

    // Check if any broken edges still exist
    const remainingBroken = Array.from(brokenEdgeIDs).filter((edgeId) =>
      connectedEdges.some((e) => e.id === edgeId),
    );

    if (remainingBroken.length === 0) {
      // Resolution complete - now apply the pending update
      if (resolutionData?.pendingHardcodedValues) {
        updateNodeData(nodeID, {
          hardcodedValues: resolutionData.pendingHardcodedValues,
        });
      }
      // setNodeResolutionMode will clean up this node's broken edges automatically
      setNodeResolutionMode(nodeID, false);
    }
  }, [
    isInResolutionMode,
    brokenEdgeIDs,
    connectedEdges,
    resolutionData,
    nodeID,
  ]);

  return {
    updateInfo,
    isInResolutionMode,
    resolutionData,
    showIncompatibilityDialog,
    setShowIncompatibilityDialog,
    handleUpdateClick,
    handleConfirmIncompatibleUpdate,
  };
}
