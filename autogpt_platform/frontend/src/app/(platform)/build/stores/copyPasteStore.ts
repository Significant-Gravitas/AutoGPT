import { create } from "zustand";
import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import { Connection, useEdgeStore } from "./edgeStore";
import { Key, storage } from "@/services/storage/local-storage";
import { useNodeStore } from "./nodeStore";

interface CopyableData {
  nodes: CustomNode[];
  connections: Connection[];
}

type CopyPasteStore = {
  copySelectedNodes: () => void;
  pasteNodes: () => void;
};

export const useCopyPasteStore = create<CopyPasteStore>((set, get) => ({
  copySelectedNodes: () => {
    const { nodes } = useNodeStore.getState();
    const { connections } = useEdgeStore.getState();

    const selectedNodes = nodes.filter((node) => node.selected);
    const selectedNodeIds = new Set(selectedNodes.map((node) => node.id));

    const selectedConnections = connections.filter(
      (conn) =>
        selectedNodeIds.has(conn.source) && selectedNodeIds.has(conn.target),
    );

    const copiedData: CopyableData = {
      nodes: selectedNodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
        },
      })),
      connections: selectedConnections,
    };

    storage.set(Key.COPIED_FLOW_DATA, JSON.stringify(copiedData));
  },

  pasteNodes: () => {
    const copiedDataString = storage.get(Key.COPIED_FLOW_DATA);
    if (!copiedDataString) return;

    const copiedData = JSON.parse(copiedDataString) as CopyableData;
    const { addNode } = useNodeStore.getState();
    const { addConnection } = useEdgeStore.getState();

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

    copiedData.connections.forEach((conn) => {
      const newSourceId = oldToNewIdMap[conn.source] ?? conn.source;
      const newTargetId = oldToNewIdMap[conn.target] ?? conn.target;

      addConnection({
        source: newSourceId,
        target: newTargetId,
        sourceHandle: conn.sourceHandle ?? "",
        targetHandle: conn.targetHandle ?? "",
      });
    });
  },
}));
