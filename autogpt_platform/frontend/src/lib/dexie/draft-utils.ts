import type { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";
import type { CustomEdge } from "@/app/(platform)/build/components/FlowEditor/edges/CustomEdge";
import isEqual from "lodash/isEqual";

export function cleanNode(node: CustomNode) {
  return {
    id: node.id,
    // Note: position is intentionally excluded to prevent draft saves when dragging nodes
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

export interface DraftDiff {
  nodes: {
    added: number;
    removed: number;
    modified: number;
  };
  edges: {
    added: number;
    removed: number;
    modified: number;
  };
}

/**
 * Calculate the diff between draft and current nodes/edges.
 * - Added: items in draft but not in current (will be restored)
 * - Removed: items in current but not in draft (will be removed if draft is loaded)
 * - Modified: items with same ID but different content
 */
export function calculateDraftDiff(
  draftNodes: CustomNode[],
  draftEdges: CustomEdge[],
  currentNodes: CustomNode[],
  currentEdges: CustomEdge[],
): DraftDiff {
  const draftNodeIds = new Set(draftNodes.map((n) => n.id));
  const currentNodeIds = new Set(currentNodes.map((n) => n.id));
  const draftEdgeIds = new Set(draftEdges.map((e) => e.id));
  const currentEdgeIds = new Set(currentEdges.map((e) => e.id));

  // Nodes diff
  const nodesAdded = draftNodes.filter((n) => !currentNodeIds.has(n.id)).length;
  const nodesRemoved = currentNodes.filter(
    (n) => !draftNodeIds.has(n.id),
  ).length;

  // Modified nodes: same ID but different content
  const draftNodeMap = new Map(draftNodes.map((n) => [n.id, cleanNode(n)]));
  const currentNodeMap = new Map(currentNodes.map((n) => [n.id, cleanNode(n)]));
  let nodesModified = 0;
  for (const [id, draftClean] of draftNodeMap) {
    const currentClean = currentNodeMap.get(id);
    if (currentClean && !isEqual(draftClean, currentClean)) {
      nodesModified++;
    }
  }

  // Edges diff
  const edgesAdded = draftEdges.filter((e) => !currentEdgeIds.has(e.id)).length;
  const edgesRemoved = currentEdges.filter(
    (e) => !draftEdgeIds.has(e.id),
  ).length;

  // Modified edges: same ID but different content
  const draftEdgeMap = new Map(draftEdges.map((e) => [e.id, cleanEdge(e)]));
  const currentEdgeMap = new Map(currentEdges.map((e) => [e.id, cleanEdge(e)]));
  let edgesModified = 0;
  for (const [id, draftClean] of draftEdgeMap) {
    const currentClean = currentEdgeMap.get(id);
    if (currentClean && !isEqual(draftClean, currentClean)) {
      edgesModified++;
    }
  }

  return {
    nodes: {
      added: nodesAdded,
      removed: nodesRemoved,
      modified: nodesModified,
    },
    edges: {
      added: edgesAdded,
      removed: edgesRemoved,
      modified: edgesModified,
    },
  };
}
