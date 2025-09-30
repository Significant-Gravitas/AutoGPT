import { create } from "zustand";
import { NodeChange, applyNodeChanges } from "@xyflow/react";
import { CustomNode } from "../components/FlowEditor/nodes/CustomNode";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { convertBlockInfoIntoCustomNodeData } from "../components/helper";

type NodeStore = {
  nodes: CustomNode[];
  nodeCounter: number;
  nodeAdvancedStates: Record<string, boolean>;
  setNodes: (nodes: CustomNode[]) => void;
  onNodesChange: (changes: NodeChange<CustomNode>[]) => void;
  addNode: (node: CustomNode) => void;
  addBlock: (block: BlockInfo) => void;
  incrementNodeCounter: () => void;
  updateNodeData: (nodeId: string, data: Partial<CustomNode["data"]>) => void;
  toggleAdvanced: (nodeId: string) => void;
  setShowAdvanced: (nodeId: string, show: boolean) => void;
  getShowAdvanced: (nodeId: string) => boolean;
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
  onNodesChange: (changes) =>
    set((state) => ({
      nodes: applyNodeChanges(changes, state.nodes),
    })),
  addNode: (node) =>
    set((state) => ({
      nodes: [...state.nodes, node],
    })),
  addBlock: (block: BlockInfo) => {
    const customNodeData = convertBlockInfoIntoCustomNodeData(block);
    get().incrementNodeCounter();
    const nodeNumber = get().nodeCounter;
    const customNode: CustomNode = {
      id: nodeNumber.toString(),
      data: customNodeData,
      type: "custom",
      position: { x: 0, y: 0 },
    };
    set((state) => ({
      nodes: [...state.nodes, customNode],
    }));
  },
  updateNodeData: (nodeId, data) =>
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, ...data } } : n,
      ),
    })),
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
}));
