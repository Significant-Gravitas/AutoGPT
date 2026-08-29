import { useNodeStore } from "../../../../stores/nodeStore";
import { useEdgeStore } from "../../../../stores/edgeStore";

export const isConnectionMade = (
  sourceBlockId: string,
  targetBlockId: string,
): boolean => {
  const edges = useEdgeStore.getState().edges;
  const nodes = useNodeStore.getState().nodes;

  const sourceNode = nodes.find((n) => n.data?.block_id === sourceBlockId);
  const targetNode = nodes.find((n) => n.data?.block_id === targetBlockId);

  if (!sourceNode || !targetNode) return false;

  return edges.some((edge) => {
    return edge.source === sourceNode.id && edge.target === targetNode.id;
  });
};
