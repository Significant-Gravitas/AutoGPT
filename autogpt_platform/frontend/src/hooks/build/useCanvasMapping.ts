import { useMemo } from "react";
import { Node, Edge } from "@xyflow/react";

interface UseCanvasMappingProps<T extends Record<string, unknown> = any> {
  nodes: Node<T>[];
  edges: Edge[];
}

export const useCanvasMapping = <T extends Record<string, unknown>>({
  nodes,
  edges,
}: UseCanvasMappingProps<T>) => {
  // Map or enhance nodes
  const mappedNodes = useMemo(
    () =>
      nodes.map((node) => ({
        ...node,
        data: { ...node.data, label: `Mapped Node - ${node.id}` },
      })),
    [nodes],
  );

  // Map or enhance edges
  const mappedEdges = useMemo(
    () =>
      edges.map((edge) => ({
        ...edge,
        style: { ...edge.style, stroke: "#00f" },
      })),
    [edges],
  );

  return { mappedNodes, mappedEdges };
};
