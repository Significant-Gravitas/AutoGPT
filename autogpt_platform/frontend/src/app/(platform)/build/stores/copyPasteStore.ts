import { create } from "zustand";
import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import { useEdgeStore } from "./edgeStore";
import { Key, storage } from "@/services/storage/local-storage";
import { useNodeStore } from "./nodeStore";
import { CustomEdge } from "../components/FlowEditor/edges/CustomEdge";

interface CopyableData {
  nodes: CustomNode[];
  edges: CustomEdge[];
}

type CopyPasteStore = {
  copySelectedNodes: () => void;
  pasteNodes: () => void;
};

export const useCopyPasteStore = create<CopyPasteStore>(() => ({
  copySelectedNodes: () => {
    const { nodes } = useNodeStore.getState();
    const { edges } = useEdgeStore.getState();

    const selectedNodes = nodes.filter((node) => node.selected);
    const selectedNodeIds = new Set(selectedNodes.map((node) => node.id));

    const selectedEdges = edges.filter(
      (edge) =>
        selectedNodeIds.has(edge.source) && selectedNodeIds.has(edge.target),
    );

    const copiedData: CopyableData = {
      nodes: selectedNodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
        },
      })),
      edges: selectedEdges,
    };

    storage.set(Key.COPIED_FLOW_DATA, JSON.stringify(copiedData));
  },

  pasteNodes: () => {
    const copiedDataString = storage.get(Key.COPIED_FLOW_DATA);
    if (!copiedDataString) return;

    const copiedData = JSON.parse(copiedDataString) as CopyableData;
    const { addNode } = useNodeStore.getState();
    const { addEdge } = useEdgeStore.getState();

    const oldToNewIdMap: Record<string, string> = {};

    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;

    copiedData.nodes.forEach((node) => {
      minX = Math.min(minX, node.position.x);
      minY = Math.min(minY, node.position.y);
      maxX = Math.max(maxX, node.position.x);
      maxY = Math.max(maxY, node.position.y);
    });

    const offsetX = 50;
    const offsetY = 50;

    useNodeStore.setState((state) => ({
      nodes: state.nodes.map((node) => ({ ...node, selected: false })),
    }));

    copiedData.nodes.forEach((node) => {
      const { incrementNodeCounter, nodeCounter } = useNodeStore.getState();
      incrementNodeCounter();
      oldToNewIdMap[node.id] = (nodeCounter + 1).toString();

      addNode({
        ...node,
        id: (nodeCounter + 1).toString(),
        position: {
          x: node.position.x + offsetX,
          y: node.position.y + offsetY,
        },
        selected: true,
      });
    });

    copiedData.edges.forEach((edge) => {
      const newSourceId = oldToNewIdMap[edge.source] ?? edge.source;
      const newTargetId = oldToNewIdMap[edge.target] ?? edge.target;

      addEdge({
        source: newSourceId,
        target: newTargetId,
        sourceHandle: edge.sourceHandle ?? "",
        targetHandle: edge.targetHandle ?? "",
      });
    });
  },
}));
