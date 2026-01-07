import type { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";
import type { CustomEdge } from "@/app/(platform)/build/components/FlowEditor/edges/CustomEdge";

export function cleanNode(node: CustomNode) {
  return {
    id: node.id,
    position: node.position,
    data: {
      hardcodedValues: node.data.hardcodedValues,
      title: node.data.title,
      block_id: node.data.block_id,
      metadata: node.data.metadata,
    },
  };
}

export function cleanEdge(edge: CustomEdge) {
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.sourceHandle,
    targetHandle: edge.targetHandle,
  };
}

export function cleanNodes(nodes: CustomNode[]) {
  return nodes.map(cleanNode);
}

export function cleanEdges(edges: CustomEdge[]) {
  return edges.map(cleanEdge);
}
