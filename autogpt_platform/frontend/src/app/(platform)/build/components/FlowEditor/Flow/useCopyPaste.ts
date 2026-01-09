import { useCallback, useEffect } from "react";
import { useReactFlow } from "@xyflow/react";
import { v4 as uuidv4 } from "uuid";
import { useNodeStore } from "../../../stores/nodeStore";
import { useEdgeStore } from "../../../stores/edgeStore";
import { CustomNode } from "../nodes/CustomNode/CustomNode";
import { CustomEdge } from "../edges/CustomEdge";
import { useToast } from "@/components/molecules/Toast/use-toast";

interface CopyableData {
  nodes: CustomNode[];
  edges: CustomEdge[];
}

const CLIPBOARD_PREFIX = "autogpt-flow-data:";

export function useCopyPaste() {
  const { getViewport } = useReactFlow();
  const { toast } = useToast();

  const handleCopyPaste = useCallback(
    (event: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const isInputField =
        activeElement?.tagName === "INPUT" ||
        activeElement?.tagName === "TEXTAREA" ||
        activeElement?.getAttribute("contenteditable") === "true";

      if (isInputField) return;

      if (event.ctrlKey || event.metaKey) {
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

          const clipboardText = `${CLIPBOARD_PREFIX}${JSON.stringify(copiedData)}`;
          navigator.clipboard
            .writeText(clipboardText)
            .then(() => {
              toast({
                title: "Copied successfully",
                description: `${selectedNodes.length} node(s) copied to clipboard`,
              });
            })
            .catch((error) => {
              console.error("Failed to copy to clipboard:", error);
            });
        }

        if (event.key === "v" || event.key === "V") {
          navigator.clipboard
            .readText()
            .then((clipboardText) => {
              if (!clipboardText.startsWith(CLIPBOARD_PREFIX)) {
                return; // Not our data, ignore
              }

              const jsonString = clipboardText.slice(CLIPBOARD_PREFIX.length);
              const copiedData = JSON.parse(jsonString) as CopyableData;
              const oldToNewIdMap: Record<string, string> = {};

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
                nodes: state.nodes.map((node) => ({
                  ...node,
                  selected: false,
                })),
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
            })
            .catch((error) => {
              console.error("Failed to read from clipboard:", error);
            });
        }
      }
    },
    [getViewport, toast],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      handleCopyPaste(event);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [handleCopyPaste]);

  return handleCopyPaste;
}
