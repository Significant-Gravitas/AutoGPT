import { useState, useCallback, useMemo, useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import {
  useNodeStore,
  NodeResolutionData,
} from "@/app/(platform)/build/stores/nodeStore";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import {
  useSubAgentUpdate,
  createUpdatedHardcodedValues,
  getBrokenEdgeIDs,
  GraphMetaLike,
} from "@/app/(platform)/build/hooks/useSubAgentUpdate";
import { GraphInputSchema, GraphOutputSchema } from "@/lib/autogpt-server-api";
import { CustomNodeData } from "../../CustomNode";

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
  const setBrokenEdgeIds = useNodeStore(
    useShallow((state) => state.setBrokenEdgeIDs),
  );
  const brokenEdgeIds = useNodeStore(
    useShallow((state) => state.brokenEdgeIDs),
  );
  const getNodeResolutionData = useNodeStore(
    useShallow((state) => state.getNodeResolutionData),
  );
  const edges = useEdgeStore(useShallow((state) => state.edges));
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

  // Get node connections
  const nodeConnections = useMemo(() => {
    return edges.filter(
      (edge) => edge.source === nodeID || edge.target === nodeID,
    );
  }, [edges, nodeID]);

  // Use the sub-agent update hook
  const updateInfo = useSubAgentUpdate(
    nodeID,
    graphID,
    graphVersion,
    currentInputSchema,
    currentOutputSchema,
    nodeConnections,
    availableSubGraphs,
  );

  const isInResolutionMode = isNodeInResolutionMode(nodeID);

  // Handle update button click
  const handleUpdateClick = useCallback(() => {
    if (!updateInfo.hasUpdate || !updateInfo.latestFlow) return;

    if (updateInfo.isCompatible) {
      // Compatible update - apply directly
      const newHardcodedValues = createUpdatedHardcodedValues(
        nodeData.hardcodedValues,
        updateInfo.latestFlow as GraphMetaLike,
      );
      updateNodeData(nodeID, { hardcodedValues: newHardcodedValues });
    } else {
      // Incompatible update - show dialog
      setShowIncompatibilityDialog(true);
    }
  }, [
    updateInfo.hasUpdate,
    updateInfo.latestFlow,
    updateInfo.isCompatible,
    nodeData.hardcodedValues,
    updateNodeData,
    nodeID,
  ]);

  // Handle confirming an incompatible update
  const handleConfirmIncompatibleUpdate = useCallback(() => {
    if (!updateInfo.latestFlow || !updateInfo.incompatibilities) return;

    const latestFlow = updateInfo.latestFlow as GraphMetaLike;

    // Get the new schemas from the latest flow
    const newInputSchema =
      (latestFlow.input_schema as Record<string, unknown>) || {};
    const newOutputSchema =
      (latestFlow.output_schema as Record<string, unknown>) || {};

    // Create the updated hardcoded values but DON'T apply them yet
    // We'll apply them when resolution is complete
    const pendingHardcodedValues = createUpdatedHardcodedValues(
      nodeData.hardcodedValues,
      latestFlow,
    );

    // Get broken edge IDs
    const brokenIds = getBrokenEdgeIDs(
      nodeConnections,
      updateInfo.incompatibilities,
      nodeID,
    );
    setBrokenEdgeIds(brokenIds);

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
  }, [
    updateInfo.latestFlow,
    updateInfo.incompatibilities,
    nodeData.hardcodedValues,
    currentInputSchema,
    currentOutputSchema,
    nodeID,
    nodeConnections,
    setBrokenEdgeIds,
    setNodeResolutionMode,
  ]);

  // Check if resolution is complete (all broken edges removed)
  const resolutionData = getNodeResolutionData(nodeID);

  // Auto-check resolution on edge changes
  useEffect(() => {
    if (!isInResolutionMode) return;

    // Check if any broken edges still exist
    const remainingBroken = Array.from(brokenEdgeIds).filter((edgeId) =>
      edges.some((e) => e.id === edgeId),
    );

    if (remainingBroken.length === 0) {
      // Resolution complete - now apply the pending update
      if (resolutionData?.pendingHardcodedValues) {
        updateNodeData(nodeID, {
          hardcodedValues: resolutionData.pendingHardcodedValues,
        });
      }
      setNodeResolutionMode(nodeID, false);
      setBrokenEdgeIds([]);
    }
  }, [
    isInResolutionMode,
    brokenEdgeIds,
    edges,
    resolutionData,
    updateNodeData,
    setNodeResolutionMode,
    nodeID,
    setBrokenEdgeIds,
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
