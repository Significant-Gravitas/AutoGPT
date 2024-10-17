import { useCallback } from "react";
import { Node, Edge, useReactFlow, useViewport } from "@xyflow/react";

export function useCopyPaste(getNextNodeId: () => string) {
  const { setNodes, addEdges, getNodes, getEdges } = useReactFlow();
  const { x, y, zoom } = useViewport();

  const handleCopyPaste = useCallback(
    (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey) {
        if (event.key === "c" || event.key === "C") {
          const selectedNodes = getNodes().filter((node) => node.selected);
          const selectedEdges = getEdges().filter((edge) => edge.selected);

          const copiedData = {
            nodes: selectedNodes.map((node) => ({
              ...node,
              data: {
                ...node.data,
                connections: [],
              },
            })),
            edges: selectedEdges,
          };

          localStorage.setItem("copiedFlowData", JSON.stringify(copiedData));
        }
        if (event.key === "v" || event.key === "V") {
          const copiedDataString = localStorage.getItem("copiedFlowData");
          if (copiedDataString) {
            const copiedData = JSON.parse(copiedDataString);
            const oldToNewIdMap: Record<string, string> = {};

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

            const pastedNodes = copiedData.nodes.map((node: Node) => {
              const newNodeId = getNextNodeId();
              oldToNewIdMap[node.id] = newNodeId;
              return {
                ...node,
                id: newNodeId,
                position: {
                  x: node.position.x + offsetX,
                  y: node.position.y + offsetY,
                },
                data: {
                  ...node.data,
                  status: undefined,
                  executionResults: undefined,
                },
              };
            });

            const pastedEdges = copiedData.edges.map((edge: Edge) => {
              const newSourceId = oldToNewIdMap[edge.source] ?? edge.source;
              const newTargetId = oldToNewIdMap[edge.target] ?? edge.target;
              return {
                ...edge,
                id: `${newSourceId}_${edge.sourceHandle}_${newTargetId}_${edge.targetHandle}_${Date.now()}`,
                source: newSourceId,
                target: newTargetId,
              };
            });

            setNodes((existingNodes) => [
              ...existingNodes.map((node) => ({ ...node, selected: false })),
              ...pastedNodes,
            ]);
            addEdges(pastedEdges);

            setNodes((nodes) => {
              return nodes.map((node) => {
                if (oldToNewIdMap[node.id]) {
                  const nodeConnections = pastedEdges
                    .filter(
                      (edge) =>
                        edge.source === node.id || edge.target === node.id,
                    )
                    .map((edge) => ({
                      edge_id: edge.id,
                      source: edge.source,
                      target: edge.target,
                      sourceHandle: edge.sourceHandle,
                      targetHandle: edge.targetHandle,
                    }));
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      connections: nodeConnections,
                    },
                  };
                }
                return node;
              });
            });
          }
        }
      }
    },
    [setNodes, addEdges, getNodes, getEdges, getNextNodeId, x, y, zoom],
  );

  return handleCopyPaste;
}
