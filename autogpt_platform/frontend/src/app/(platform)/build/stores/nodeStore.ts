import { create } from "zustand";
import { NodeChange, XYPosition, applyNodeChanges } from "@xyflow/react";
import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import {
  convertBlockInfoIntoCustomNodeData,
  findFreePosition,
} from "../components/helper";
import { Node } from "@/app/api/__generated__/models/node";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import { useHistoryStore } from "./historyStore";
import { useEdgeStore } from "./edgeStore";
import { BlockUIType } from "../components/types";
import { pruneEmptyValues } from "@/lib/utils";
import {
  ensurePathExists,
  parseHandleIdToPath,
} from "@/components/renderers/InputRenderer/helpers";

// Minimum movement (in pixels) required before logging position change to history
// Prevents spamming history with small movements when clicking on inputs inside blocks
const MINIMUM_MOVE_BEFORE_LOG = 50;

// Track initial positions when drag starts (outside store to avoid re-renders)
const dragStartPositions: Record<string, XYPosition> = {};

type NodeStore = {
  nodes: CustomNode[];
  nodeCounter: number;
  nodeAdvancedStates: Record<string, boolean>;
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

  updateNodeExecutionResult: (
    nodeId: string,
    result: NodeExecutionResult,
  ) => void;
  getNodeExecutionResult: (nodeId: string) => NodeExecutionResult | undefined;
  getNodeBlockUIType: (nodeId: string) => BlockUIType;
  hasWebhookNodes: () => boolean;

  updateNodeErrors: (nodeId: string, errors: { [key: string]: string }) => void;
  clearNodeErrors: (nodeId: string) => void;
  getNodeErrors: (nodeId: string) => { [key: string]: string } | undefined;
  setNodeErrorsForBackendId: (
    backendId: string,
    errors: { [key: string]: string },
  ) => void;
  clearAllNodeErrors: () => void; // Add this

  syncHardcodedValuesWithHandleIds: (nodeId: string) => void;
};

export const useNodeStore = create<NodeStore>((set, get) => ({
  nodes: [],
  setNodes: (nodes) => set({ nodes }),
  nodeCounter: 0,
  nodeAdvancedStates: {},
  incrementNodeCounter: () =>
    set((state) => ({
      nodeCounter: state.nodeCounter + 1,
    })),
  onNodesChange: (changes) => {
    const prevState = {
      nodes: get().nodes,
      edges: useEdgeStore.getState().edges,
    };

    // Track initial positions when drag starts
    changes.forEach((change) => {
      if (change.type === "position" && change.dragging === true) {
        if (!dragStartPositions[change.id]) {
          const node = get().nodes.find((n) => n.id === change.id);
          if (node) {
            dragStartPositions[change.id] = { ...node.position };
          }
        }
      }
    });

    // Check if we should track this change in history
    let shouldTrack = changes.some(
      (change) => change.type === "remove" || change.type === "add",
    );

    // For position changes, only track if movement exceeds threshold
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
            }
          }
          // Clean up tracked position after drag ends
          delete dragStartPositions[change.id];
        }
      });
    }

    set((state) => ({
      nodes: applyNodeChanges(changes, state.nodes),
    }));

    if (shouldTrack) {
      useHistoryStore.getState().pushState(prevState);
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
    return customNode;
  },
  updateNodeData: (nodeId, data) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, ...data } } : n,
      ),
    }));

    const newState = {
      nodes: get().nodes,
      edges: useEdgeStore.getState().edges,
    };

    useHistoryStore.getState().pushState(newState);
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

  updateNodeExecutionResult: (nodeId: string, result: NodeExecutionResult) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, nodeExecutionResult: result } }
          : n,
      ),
    }));
  },
  getNodeExecutionResult: (nodeId: string) => {
    return get().nodes.find((n) => n.id === nodeId)?.data?.nodeExecutionResult;
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
}));
