import { create } from "zustand";
import { NodeChange, applyNodeChanges } from "@xyflow/react";
import { CustomNode } from "../FlowEditor/CustomNode/CustomNode";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { convertBlockInfoIntoCustomNodeData } from "../helper";

type NodeStore = {
  nodes: CustomNode[];
  nodeCounter: number;
  setNodes: (nodes: CustomNode[]) => void;
  onNodesChange: (changes: NodeChange<CustomNode>[]) => void;
  addNode: (node: CustomNode) => void;
  addBlock: (block: BlockInfo) => void;
  incrementNodeCounter: () => void;
};

export const useNodeStore = create<NodeStore>((set, get) => ({
  nodes: [],
  setNodes: (nodes) => set({ nodes }),
  nodeCounter: 0,
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
}));
