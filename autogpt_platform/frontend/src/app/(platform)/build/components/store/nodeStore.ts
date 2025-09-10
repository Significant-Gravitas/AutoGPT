import { create } from "zustand";
import { Node, NodeChange, applyNodeChanges } from "@xyflow/react";
import { CustomNode } from "../Flow/CustomNode/CustomNode";

type NodeStore = {
  nodes: CustomNode[];
  setNodes: (nodes: CustomNode[]) => void;
  onNodesChange: (changes: NodeChange<CustomNode>[]) => void;
  addNode: (node: CustomNode) => void;
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
}));
