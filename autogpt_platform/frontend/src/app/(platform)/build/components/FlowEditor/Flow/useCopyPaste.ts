import { useCallback } from "react";
import { useReactFlow } from "@xyflow/react";
import { Key, storage } from "@/services/storage/local-storage";
import { v4 as uuidv4 } from "uuid";
import { useNodeStore } from "../../../stores/nodeStore";
import { useEdgeStore } from "../../../stores/edgeStore";
import { CustomNode } from "../nodes/CustomNode/CustomNode";
import { CustomEdge } from "../edges/CustomEdge";

interface CopyableData {
  nodes: CustomNode[];
  edges: CustomEdge[];
}

export function useCopyPaste() {
  // Only use useReactFlow for viewport (not managed by stores)
  const { getViewport } = useReactFlow();

  const handleCopyPaste = useCallback(
    (event: KeyboardEvent) => {
      // Prevent copy/paste if any modal is open or if the focus is on an input element
      const activeElement = document.activeElement;
      const isInputField =
        activeElement?.tagName === "INPUT" ||
        activeElement?.tagName === "TEXTAREA" ||
        activeElement?.getAttribute("contenteditable") === "true";

      if (isInputField) return;

      if (event.ctrlKey || event.metaKey) {
        // COPY: Ctrl+C or Cmd+C
        if (event.key === "c" || event.key === "C") {
          const { nodes } = useNodeStore.getState();
          const { edges } = useEdgeStore.getState();

          const selectedNodes = nodes.filter((node) => node.selected);
          const selectedNodeIds = new Set(selectedNodes.map((node) => node.id));

          // Copy edges where both source and target nodes are selected
          const selectedEdges = edges.filter(
            (edge) =>
              selectedNodeIds.has(edge.source) &&
              selectedNodeIds.has(edge.target),
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
        }

        // PASTE: Ctrl+V or Cmd+V
        if (event.key === "v" || event.key === "V") {
          const copiedDataString = storage.get(Key.COPIED_FLOW_DATA);
          if (copiedDataString) {
            const copiedData = JSON.parse(copiedDataString) as CopyableData;
            const oldToNewIdMap: Record<string, string> = {};

            // Get fresh viewport values at paste time to ensure correct positioning
            const { x, y, zoom } = getViewport();
            const viewportCenter = {
              x: (window.innerWidth / 2 - x) / zoom,
              y: (window.innerHeight / 2 - y) / zoom,
            };

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

            const offsetX = viewportCenter.x - (minX + maxX) / 2;
            const offsetY = viewportCenter.y - (minY + maxY) / 2;

            // Deselect existing nodes first
            useNodeStore.setState((state) => ({
              nodes: state.nodes.map((node) => ({ ...node, selected: false })),
            }));

            // Create and add new nodes with UNIQUE IDs using UUID
            copiedData.nodes.forEach((node) => {
              const newNodeId = uuidv4();
              oldToNewIdMap[node.id] = newNodeId;

              const newNode: CustomNode = {
                ...node,
                id: newNodeId,
                selected: true,
                position: {
                  x: node.position.x + offsetX,
                  y: node.position.y + offsetY,
                },
              };

              useNodeStore.getState().addNode(newNode);
            });

            // Add edges with updated source/target IDs
            const { addEdge } = useEdgeStore.getState();
            copiedData.edges.forEach((edge) => {
              const newSourceId = oldToNewIdMap[edge.source] ?? edge.source;
              const newTargetId = oldToNewIdMap[edge.target] ?? edge.target;

              addEdge({
                source: newSourceId,
                target: newTargetId,
                sourceHandle: edge.sourceHandle ?? "",
                targetHandle: edge.targetHandle ?? "",
                data: {
                  ...edge.data,
                },
              });
            });
          }
        }
      }
    },
    [getViewport],
  );

  return handleCopyPaste;
}
