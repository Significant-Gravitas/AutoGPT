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
    const shouldTrack = changes.some(
      (change) =>
        change.type === "remove" ||
        change.type === "add" ||
        (change.type === "position" && change.dragging === false),
    );
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
            width: node.data.uiType === BlockUIType.NOTE ? 300 : 500,
            height: 400,
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
      input_default: node.data.hardcodedValues,
      metadata: {
        // TODO: Add more metadata
        position: node.position,
        customized_name: node.data.metadata?.customized_name,
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
}));
