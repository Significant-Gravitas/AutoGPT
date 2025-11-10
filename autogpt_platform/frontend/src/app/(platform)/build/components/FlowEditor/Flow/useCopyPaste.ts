import { useCallback } from "react";
import { Node, Edge, useReactFlow } from "@xyflow/react";
import { Key, storage } from "@/services/storage/local-storage";
import { v4 as uuidv4 } from "uuid";

interface CopyableData {
  nodes: Node[];
  edges: Edge[];
}

export function useCopyPaste() {
  const { setNodes, addEdges, getNodes, getEdges, getViewport } =
    useReactFlow();

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
          const selectedNodes = getNodes().filter((node) => node.selected);
          const selectedNodeIds = new Set(selectedNodes.map((node) => node.id));

          // Only copy edges where both source and target nodes are selected
          const selectedEdges = getEdges().filter(
            (edge) =>
              edge.selected &&
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
            copiedData.nodes.forEach((node: Node) => {
              minX = Math.min(minX, node.position.x);
              minY = Math.min(minY, node.position.y);
              maxX = Math.max(maxX, node.position.x);
              maxY = Math.max(maxY, node.position.y);
            });

            const offsetX = viewportCenter.x - (minX + maxX) / 2;
            const offsetY = viewportCenter.y - (minY + maxY) / 2;

            // Create new nodes with UNIQUE IDs using UUID
            const pastedNodes = copiedData.nodes.map((node: Node) => {
              const newNodeId = uuidv4(); // Generate unique UUID for each node
              oldToNewIdMap[node.id] = newNodeId;
              return {
                ...node,
                id: newNodeId, // Assign the new unique ID
                selected: true, // Select the pasted nodes
                position: {
                  x: node.position.x + offsetX,
                  y: node.position.y + offsetY,
                },
                data: {
                  ...node.data,
                  backend_id: undefined, // Clear backend_id so the new node.id is used when saving
                  status: undefined, // Clear execution status
                  nodeExecutionResult: undefined, // Clear execution results
                },
              };
            });

            // Create new edges with updated source/target IDs
            const pastedEdges = copiedData.edges.map((edge) => {
              const newSourceId = oldToNewIdMap[edge.source] ?? edge.source;
              const newTargetId = oldToNewIdMap[edge.target] ?? edge.target;
              return {
                ...edge,
                id: `${newSourceId}_${edge.sourceHandle}_${newTargetId}_${edge.targetHandle}_${Date.now()}`,
                source: newSourceId,
                target: newTargetId,
              };
            });

            // Deselect existing nodes and add pasted nodes
            setNodes((existingNodes) => [
              ...existingNodes.map((node) => ({ ...node, selected: false })),
              ...pastedNodes,
            ]);
            addEdges(pastedEdges);
          }
        }
      }
    },
    [setNodes, addEdges, getNodes, getEdges, getViewport],
  );

  return handleCopyPaste;
}
