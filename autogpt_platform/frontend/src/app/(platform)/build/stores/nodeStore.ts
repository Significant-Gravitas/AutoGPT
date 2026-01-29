import { create } from "zustand";
import { NodeChange, XYPosition, applyNodeChanges } from "@xyflow/react";
import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import { CustomEdge } from "../components/FlowEditor/edges/CustomEdge";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import {
  convertBlockInfoIntoCustomNodeData,
  findFreePosition,
} from "../components/helper";
import { Node } from "@/app/api/__generated__/models/node";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import { NodeExecutionResultInputData } from "@/app/api/__generated__/models/nodeExecutionResultInputData";
import { NodeExecutionResultOutputData } from "@/app/api/__generated__/models/nodeExecutionResultOutputData";
import { useHistoryStore } from "./historyStore";
import { useEdgeStore } from "./edgeStore";
import { BlockUIType } from "../components/types";
import { pruneEmptyValues } from "@/lib/utils";
import {
  ensurePathExists,
  parseHandleIdToPath,
} from "@/components/renderers/InputRenderer/helpers";
import { accumulateExecutionData } from "./helpers";
import { NodeResolutionData } from "./types";

const MINIMUM_MOVE_BEFORE_LOG = 50;
const dragStartPositions: Record<string, XYPosition> = {};

let dragStartState: { nodes: CustomNode[]; edges: CustomEdge[] } | null = null;

type NodeStore = {
  nodes: CustomNode[];
  nodeCounter: number;
  setNodeCounter: (nodeCounter: number) => void;
  nodeAdvancedStates: Record<string, boolean>;

  latestNodeInputData: Record<string, NodeExecutionResultInputData | undefined>;
  latestNodeOutputData: Record<
    string,
    NodeExecutionResultOutputData | undefined
  >;
  accumulatedNodeInputData: Record<string, Record<string, unknown[]>>;
  accumulatedNodeOutputData: Record<string, Record<string, unknown[]>>;

  setNodes: (nodes: CustomNode[]) => void;
  onNodesChange: (changes: NodeChange<CustomNode>[]) => void;
  addNode: (node: CustomNode) => void;
  addBlock: (
    block: BlockInfo,
    hardcodedValues?: Record<string, any>,
    position?: XYPosition,
  ) => CustomNode;
  incrementNodeCounter: () => void;
  updateNodeData: (nodeId: string, data: Partial<CustomNode["data"]>) => void;
  toggleAdvanced: (nodeId: string) => void;
  setShowAdvanced: (nodeId: string, show: boolean) => void;
  getShowAdvanced: (nodeId: string) => boolean;
  addNodes: (nodes: CustomNode[]) => void;
  getHardCodedValues: (nodeId: string) => Record<string, any>;
  convertCustomNodeToBackendNode: (node: CustomNode) => Node;
  getBackendNodes: () => Node[];

  updateNodeStatus: (nodeId: string, status: AgentExecutionStatus) => void;
  getNodeStatus: (nodeId: string) => AgentExecutionStatus | undefined;
  cleanNodesStatuses: () => void;

  updateNodeExecutionResult: (
    nodeId: string,
    result: NodeExecutionResult,
  ) => void;
  getNodeExecutionResults: (nodeId: string) => NodeExecutionResult[];
  getLatestNodeInputData: (
    nodeId: string,
  ) => NodeExecutionResultInputData | undefined;
  getLatestNodeOutputData: (
    nodeId: string,
  ) => NodeExecutionResultOutputData | undefined;
  getAccumulatedNodeInputData: (nodeId: string) => Record<string, unknown[]>;
  getAccumulatedNodeOutputData: (nodeId: string) => Record<string, unknown[]>;
  getLatestNodeExecutionResult: (
    nodeId: string,
  ) => NodeExecutionResult | undefined;
  clearAllNodeExecutionResults: () => void;

  getNodeBlockUIType: (nodeId: string) => BlockUIType;
  hasWebhookNodes: () => boolean;

  updateNodeErrors: (nodeId: string, errors: { [key: string]: string }) => void;
  clearNodeErrors: (nodeId: string) => void;
  getNodeErrors: (nodeId: string) => { [key: string]: string } | undefined;
  setNodeErrorsForBackendId: (
    backendId: string,
    errors: { [key: string]: string },
  ) => void;

  syncHardcodedValuesWithHandleIds: (nodeId: string) => void;

  setCredentialsOptional: (nodeId: string, optional: boolean) => void;
  clearAllNodeErrors: () => void;

  nodesInResolutionMode: Set<string>;
  brokenEdgeIDs: Map<string, Set<string>>;
  nodeResolutionData: Map<string, NodeResolutionData>;
  setNodeResolutionMode: (
    nodeID: string,
    inResolution: boolean,
    resolutionData?: NodeResolutionData,
  ) => void;
  isNodeInResolutionMode: (nodeID: string) => boolean;
  getNodeResolutionData: (nodeID: string) => NodeResolutionData | undefined;
  setBrokenEdgeIDs: (nodeID: string, edgeIDs: string[]) => void;
  removeBrokenEdgeID: (nodeID: string, edgeID: string) => void;
  isEdgeBroken: (edgeID: string) => boolean;
  clearResolutionState: () => void;

  isInputBroken: (nodeID: string, handleID: string) => boolean;
  getInputTypeMismatch: (
    nodeID: string,
    handleID: string,
  ) => string | undefined;
};

export const useNodeStore = create<NodeStore>((set, get) => ({
  nodes: [],
  setNodes: (nodes) => set({ nodes }),
  nodeCounter: 0,
  setNodeCounter: (nodeCounter) => set({ nodeCounter }),
  nodeAdvancedStates: {},
  latestNodeInputData: {},
  latestNodeOutputData: {},
  accumulatedNodeInputData: {},
  accumulatedNodeOutputData: {},
  incrementNodeCounter: () =>
    set((state) => ({
      nodeCounter: state.nodeCounter + 1,
    })),
  onNodesChange: (changes) => {
    changes.forEach((change) => {
      if (change.type === "position" && change.dragging === true) {
        if (!dragStartState) {
          const currentNodes = get().nodes;
          const currentEdges = useEdgeStore.getState().edges;
          dragStartState = {
            nodes: currentNodes.map((n) => ({
              ...n,
              position: { ...n.position },
              data: { ...n.data },
            })),
            edges: currentEdges.map((e) => ({ ...e })),
          };
        }
        if (!dragStartPositions[change.id]) {
          const node = get().nodes.find((n) => n.id === change.id);
          if (node) {
            dragStartPositions[change.id] = { ...node.position };
          }
        }
      }
    });

    let shouldTrack = changes.some((change) => change.type === "remove");
    let stateToTrack: { nodes: CustomNode[]; edges: CustomEdge[] } | null =
      null;

    if (shouldTrack) {
      stateToTrack = {
        nodes: get().nodes,
        edges: useEdgeStore.getState().edges,
      };
    }

    if (!shouldTrack) {
      changes.forEach((change) => {
        if (change.type === "position" && change.dragging === false) {
          const startPos = dragStartPositions[change.id];
          if (startPos && change.position) {
            const distanceMoved = Math.sqrt(
              Math.pow(change.position.x - startPos.x, 2) +
                Math.pow(change.position.y - startPos.y, 2),
            );
            if (distanceMoved > MINIMUM_MOVE_BEFORE_LOG) {
              shouldTrack = true;
              stateToTrack = dragStartState;
            }
          }
          delete dragStartPositions[change.id];
        }
      });
      if (Object.keys(dragStartPositions).length === 0) {
        dragStartState = null;
      }
    }

    set((state) => ({
      nodes: applyNodeChanges(changes, state.nodes),
    }));

    if (shouldTrack && stateToTrack) {
      useHistoryStore.getState().pushState(stateToTrack);
    }
  },

  addNode: (node) => {
    set((state) => ({
      nodes: [...state.nodes, node],
    }));
  },
  addBlock: (
    block: BlockInfo,
    hardcodedValues?: Record<string, any>,
    position?: XYPosition,
  ) => {
    const prevState = {
      nodes: get().nodes,
      edges: useEdgeStore.getState().edges,
    };

    const customNodeData = convertBlockInfoIntoCustomNodeData(
      block,
      hardcodedValues,
    );
    get().incrementNodeCounter();
    const nodeNumber = get().nodeCounter;

    const nodePosition =
      position ||
      findFreePosition(
        get().nodes.map((node) => ({
          position: node.position,
          measured: {
            width:
              node.width ??
              node.measured?.width ??
              (node.data.uiType === BlockUIType.NOTE ? 300 : 500),
            height: node.height ?? node.measured?.height ?? 400,
          },
        })),
        block.uiType === BlockUIType.NOTE ? 300 : 400,
        30,
      );

    const customNode: CustomNode = {
      id: nodeNumber.toString(),
      data: customNodeData,
      type: "custom",
      position: nodePosition,
    };
    set((state) => ({
      nodes: [...state.nodes, customNode],
    }));

    useHistoryStore.getState().pushState(prevState);

    return customNode;
  },
  updateNodeData: (nodeId, data) => {
    const prevState = {
      nodes: get().nodes,
      edges: useEdgeStore.getState().edges,
    };

    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, ...data } } : n,
      ),
    }));

    useHistoryStore.getState().pushState(prevState);
  },
  toggleAdvanced: (nodeId: string) =>
    set((state) => ({
      nodeAdvancedStates: {
        ...state.nodeAdvancedStates,
        [nodeId]: !state.nodeAdvancedStates[nodeId],
      },
    })),
  setShowAdvanced: (nodeId: string, show: boolean) =>
    set((state) => ({
      nodeAdvancedStates: {
        ...state.nodeAdvancedStates,
        [nodeId]: show,
      },
    })),
  getShowAdvanced: (nodeId: string) =>
    get().nodeAdvancedStates[nodeId] || false,
  addNodes: (nodes: CustomNode[]) => {
    nodes.forEach((node) => {
      get().addNode(node);
    });
  },
  getHardCodedValues: (nodeId: string) => {
    return (
      get().nodes.find((n) => n.id === nodeId)?.data?.hardcodedValues || {}
    );
  },
  convertCustomNodeToBackendNode: (node: CustomNode) => {
    return {
      id: node.id,
      block_id: node.data.block_id,
      input_default: pruneEmptyValues(node.data.hardcodedValues),
      metadata: {
        position: node.position,
        ...(node.data.metadata?.customized_name !== undefined && {
          customized_name: node.data.metadata.customized_name,
        }),
        ...(node.data.metadata?.credentials_optional !== undefined && {
          credentials_optional: node.data.metadata.credentials_optional,
        }),
      },
    };
  },
  getBackendNodes: () => {
    return get().nodes.map((node) =>
      get().convertCustomNodeToBackendNode(node),
    );
  },
  updateNodeStatus: (nodeId: string, status: AgentExecutionStatus) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, status } } : n,
      ),
    }));
  },
  getNodeStatus: (nodeId: string) => {
    return get().nodes.find((n) => n.id === nodeId)?.data?.status;
  },

  cleanNodesStatuses: () => {
    set((state) => ({
      nodes: state.nodes.map((n) => ({
        ...n,
        data: { ...n.data, status: undefined },
      })),
    }));
  },

  updateNodeExecutionResult: (nodeId: string, result: NodeExecutionResult) => {
    set((state) => {
      let latestNodeInputData = state.latestNodeInputData;
      let latestNodeOutputData = state.latestNodeOutputData;
      let accumulatedNodeInputData = state.accumulatedNodeInputData;
      let accumulatedNodeOutputData = state.accumulatedNodeOutputData;

      const nodes = state.nodes.map((n) => {
        if (n.id !== nodeId) return n;

        const existingResults = n.data.nodeExecutionResults || [];
        const duplicateIndex = existingResults.findIndex(
          (r) => r.node_exec_id === result.node_exec_id,
        );

        if (duplicateIndex !== -1) {
          const oldResult = existingResults[duplicateIndex];
          const inputDataChanged =
            JSON.stringify(oldResult.input_data) !==
            JSON.stringify(result.input_data);
          const outputDataChanged =
            JSON.stringify(oldResult.output_data) !==
            JSON.stringify(result.output_data);

          if (!inputDataChanged && !outputDataChanged) {
            return n;
          }

          const updatedResults = [...existingResults];
          updatedResults[duplicateIndex] = result;

          const recomputedAccumulatedInput = updatedResults.reduce(
            (acc, r) => accumulateExecutionData(acc, r.input_data),
            {} as Record<string, unknown[]>,
          );
          const recomputedAccumulatedOutput = updatedResults.reduce(
            (acc, r) => accumulateExecutionData(acc, r.output_data),
            {} as Record<string, unknown[]>,
          );

          const mostRecentResult = updatedResults[updatedResults.length - 1];
          latestNodeInputData = {
            ...latestNodeInputData,
            [nodeId]: mostRecentResult.input_data,
          };
          latestNodeOutputData = {
            ...latestNodeOutputData,
            [nodeId]: mostRecentResult.output_data,
          };

          accumulatedNodeInputData = {
            ...accumulatedNodeInputData,
            [nodeId]: recomputedAccumulatedInput,
          };
          accumulatedNodeOutputData = {
            ...accumulatedNodeOutputData,
            [nodeId]: recomputedAccumulatedOutput,
          };

          return {
            ...n,
            data: {
              ...n.data,
              nodeExecutionResults: updatedResults,
            },
          };
        }

        accumulatedNodeInputData = {
          ...accumulatedNodeInputData,
          [nodeId]: accumulateExecutionData(
            accumulatedNodeInputData[nodeId] || {},
            result.input_data,
          ),
        };
        accumulatedNodeOutputData = {
          ...accumulatedNodeOutputData,
          [nodeId]: accumulateExecutionData(
            accumulatedNodeOutputData[nodeId] || {},
            result.output_data,
          ),
        };

        latestNodeInputData = {
          ...latestNodeInputData,
          [nodeId]: result.input_data,
        };
        latestNodeOutputData = {
          ...latestNodeOutputData,
          [nodeId]: result.output_data,
        };

        return {
          ...n,
          data: {
            ...n.data,
            nodeExecutionResults: [...existingResults, result],
          },
        };
      });

      return {
        nodes,
        latestNodeInputData,
        latestNodeOutputData,
        accumulatedNodeInputData,
        accumulatedNodeOutputData,
      };
    });
  },
  getNodeExecutionResults: (nodeId: string) => {
    return (
      get().nodes.find((n) => n.id === nodeId)?.data?.nodeExecutionResults || []
    );
  },
  getLatestNodeInputData: (nodeId: string) => {
    return get().latestNodeInputData[nodeId];
  },
  getLatestNodeOutputData: (nodeId: string) => {
    return get().latestNodeOutputData[nodeId];
  },
  getAccumulatedNodeInputData: (nodeId: string) => {
    return get().accumulatedNodeInputData[nodeId] || {};
  },
  getAccumulatedNodeOutputData: (nodeId: string) => {
    return get().accumulatedNodeOutputData[nodeId] || {};
  },
  getLatestNodeExecutionResult: (nodeId: string) => {
    const results =
      get().nodes.find((n) => n.id === nodeId)?.data?.nodeExecutionResults ||
      [];
    return results.length > 0 ? results[results.length - 1] : undefined;
  },
  clearAllNodeExecutionResults: () => {
    set((state) => ({
      nodes: state.nodes.map((n) => ({
        ...n,
        data: {
          ...n.data,
          nodeExecutionResults: [],
        },
      })),
      latestNodeInputData: {},
      latestNodeOutputData: {},
      accumulatedNodeInputData: {},
      accumulatedNodeOutputData: {},
    }));
  },
  getNodeBlockUIType: (nodeId: string) => {
    return (
      get().nodes.find((n) => n.id === nodeId)?.data?.uiType ??
      BlockUIType.STANDARD
    );
  },
  hasWebhookNodes: () => {
    return get().nodes.some((n) =>
      [BlockUIType.WEBHOOK, BlockUIType.WEBHOOK_MANUAL].includes(n.data.uiType),
    );
  },

  updateNodeErrors: (nodeId: string, errors: { [key: string]: string }) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, errors } } : n,
      ),
    }));
  },

  clearNodeErrors: (nodeId: string) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, errors: undefined } } : n,
      ),
    }));
  },

  getNodeErrors: (nodeId: string) => {
    return get().nodes.find((n) => n.id === nodeId)?.data?.errors;
  },

  setNodeErrorsForBackendId: (
    backendId: string,
    errors: { [key: string]: string },
  ) => {
    set((state) => ({
      nodes: state.nodes.map((n) => {
        // Match by backend_id if nodes have it, or by id
        const matches =
          n.data.metadata?.backend_id === backendId || n.id === backendId;
        return matches ? { ...n, data: { ...n.data, errors } } : n;
      }),
    }));
  },

  clearAllNodeErrors: () => {
    set((state) => ({
      nodes: state.nodes.map((n) => ({
        ...n,
        data: { ...n.data, errors: undefined },
      })),
    }));
  },

  syncHardcodedValuesWithHandleIds: (nodeId: string) => {
    const node = get().nodes.find((n) => n.id === nodeId);
    if (!node) return;

    const handleIds = useEdgeStore.getState().getAllHandleIdsOfANode(nodeId);
    const additionalHandles = handleIds.filter((h) => h.includes("_#_"));

    if (additionalHandles.length === 0) return;

    const hardcodedValues = JSON.parse(
      JSON.stringify(node.data.hardcodedValues || {}),
    );

    let modified = false;

    additionalHandles.forEach((handleId) => {
      const segments = parseHandleIdToPath(handleId);
      if (ensurePathExists(hardcodedValues, segments)) {
        modified = true;
      }
    });

    if (modified) {
      set((state) => ({
        nodes: state.nodes.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, hardcodedValues } } : n,
        ),
      }));
    }
  },

  setCredentialsOptional: (nodeId: string, optional: boolean) => {
    const prevState = {
      nodes: get().nodes,
      edges: useEdgeStore.getState().edges,
    };

    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId
          ? {
              ...n,
              data: {
                ...n.data,
                metadata: {
                  ...n.data.metadata,
                  credentials_optional: optional,
                },
              },
            }
          : n,
      ),
    }));

    useHistoryStore.getState().pushState(prevState);
  },

  // Sub-agent resolution mode state
  nodesInResolutionMode: new Set<string>(),
  brokenEdgeIDs: new Map<string, Set<string>>(),
  nodeResolutionData: new Map<string, NodeResolutionData>(),

  setNodeResolutionMode: (
    nodeID: string,
    inResolution: boolean,
    resolutionData?: NodeResolutionData,
  ) => {
    set((state) => {
      const newNodesSet = new Set(state.nodesInResolutionMode);
      const newResolutionDataMap = new Map(state.nodeResolutionData);
      const newBrokenEdgeIDs = new Map(state.brokenEdgeIDs);

      if (inResolution) {
        newNodesSet.add(nodeID);
        if (resolutionData) {
          newResolutionDataMap.set(nodeID, resolutionData);
        }
      } else {
        newNodesSet.delete(nodeID);
        newResolutionDataMap.delete(nodeID);
        newBrokenEdgeIDs.delete(nodeID); // Clean up broken edges when exiting resolution mode
      }

      return {
        nodesInResolutionMode: newNodesSet,
        nodeResolutionData: newResolutionDataMap,
        brokenEdgeIDs: newBrokenEdgeIDs,
      };
    });
  },

  isNodeInResolutionMode: (nodeID: string) => {
    return get().nodesInResolutionMode.has(nodeID);
  },

  getNodeResolutionData: (nodeID: string) => {
    return get().nodeResolutionData.get(nodeID);
  },

  setBrokenEdgeIDs: (nodeID: string, edgeIDs: string[]) => {
    set((state) => {
      const newMap = new Map(state.brokenEdgeIDs);
      newMap.set(nodeID, new Set(edgeIDs));
      return { brokenEdgeIDs: newMap };
    });
  },

  removeBrokenEdgeID: (nodeID: string, edgeID: string) => {
    set((state) => {
      const newMap = new Map(state.brokenEdgeIDs);
      const nodeSet = new Set(newMap.get(nodeID) || []);
      nodeSet.delete(edgeID);
      newMap.set(nodeID, nodeSet);
      return { brokenEdgeIDs: newMap };
    });
  },

  isEdgeBroken: (edgeID: string) => {
    // Check across all nodes
    const brokenEdgeIDs = get().brokenEdgeIDs;
    for (const edgeSet of brokenEdgeIDs.values()) {
      if (edgeSet.has(edgeID)) {
        return true;
      }
    }
    return false;
  },

  clearResolutionState: () => {
    set({
      nodesInResolutionMode: new Set<string>(),
      brokenEdgeIDs: new Map<string, Set<string>>(),
      nodeResolutionData: new Map<string, NodeResolutionData>(),
    });
  },

  // Helper functions for input renderers
  isInputBroken: (nodeID: string, handleID: string) => {
    const resolutionData = get().nodeResolutionData.get(nodeID);
    if (!resolutionData) return false;
    return resolutionData.incompatibilities.missingInputs.includes(handleID);
  },

  getInputTypeMismatch: (nodeID: string, handleID: string) => {
    const resolutionData = get().nodeResolutionData.get(nodeID);
    if (!resolutionData) return undefined;
    const mismatch = resolutionData.incompatibilities.inputTypeMismatches.find(
      (m) => m.name === handleID,
    );
    return mismatch?.newType;
  },
}));
