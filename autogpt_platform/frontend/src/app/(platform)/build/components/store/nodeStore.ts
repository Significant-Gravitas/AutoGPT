import { create } from "zustand";
import { Node, NodeChange, applyNodeChanges } from "@xyflow/react";
import { CustomNode } from "../FlowEditor/CustomNode/CustomNode";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { convertBlockInfoIntoCustomNodeData } from "../helper";

type NodeStore = {
  nodes: CustomNode[];
  setNodes: (nodes: CustomNode[]) => void;
  onNodesChange: (changes: NodeChange<CustomNode>[]) => void;
  addNode: (node: CustomNode) => void;
  addBlock: (block: BlockInfo) => void;
};

export const useNodeStore = create<NodeStore>((set) => ({
  nodes: [],
  setNodes: (nodes) => set({ nodes }),
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
    const customNode: CustomNode = {
      data: customNodeData,
      id: block.id,
      type: "custom",
      position: { x: 0, y: 0 },
    };
    set((state) => ({
      nodes: [...state.nodes, customNode],
    }));
  },
}));
