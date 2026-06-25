import { Connection } from "@xyflow/react";
import { Block, BlockUIType, Link } from "./types";
import { Graph } from "@/app/api/__generated__/models/graph";

export function removeAgentInputBlockValues(graph: Graph, blocks: Block[]) {
  const inputBlocks = graph.nodes?.filter(
    (node) => blocks.find((b) => b.id === node.block_id)?.uiType === BlockUIType.INPUT,
  );

  const modifiedNodes = graph.nodes?.map((node) => {
    if (inputBlocks?.find((inputNode) => inputNode.id === node.id)) {
      return {
        ...node,
        input_default: {
          ...node.input_default,
          value: "",
        },
      };
    }
    return node;
  });

  return {
    ...graph,
    nodes: modifiedNodes,
  };
}

export function formatEdgeID(conn: Link | Connection): string {
  if ("sink_id" in conn) {
    return `${conn.source_id}_${conn.source_name}_${conn.sink_id}_${conn.sink_name}`;
  } else {
    return `${conn.source}_${conn.sourceHandle}_${conn.target}_${conn.targetHandle}`;
  }
}

/** Sanitizes a graph object in place so it can "safely" be imported into the system.
 *
 * **⚠️ Note:** not an actual safety feature, just intended to make the import UX more reliable.
 * Backend handles credential validation in validate_graph() and on_graph_activate().
 * updateBlockIDs() and removeCredentials() have been removed:
 * - updateBlockIDs was 8+ months past its removal date (2025-10-01)
 * - removeCredentials was too aggressive, stripping legitimate input data
 */
export function sanitizeImportedGraph(_graph: Graph): void {
  // Intentionally no-op: importing should not mutate user-provided graph
  // payloads. Backend validation is the source of truth.
}

/**
 * Validates graph structure before submission.
 * Returns a list of error messages (empty = valid).
 */
export function validateGraphStructure(graph: Graph): string[] {
  const errors: string[] = [];

  if (!Array.isArray(graph.nodes)) {
    errors.push("Graph 'nodes' must be an array");
    return errors;
  }
  if (!Array.isArray(graph.links)) {
    errors.push("Graph 'links' must be an array");
    return errors;
  }

  if (!graph.nodes.length) {
    errors.push("Graph has no nodes");
    return errors;
  }

  graph.nodes.forEach((node, i) => {
    if (!node.block_id) {
      errors.push(`Node #${i + 1}: missing block_id`);
    }
    if (!node.id) {
      errors.push(`Node #${i + 1}: missing node id`);
    }
  });

  graph.links.forEach((link, i) => {
    if (!link.source_id) errors.push(`Link #${i + 1}: missing source_id`);
    if (!link.sink_id) errors.push(`Link #${i + 1}: missing sink_id`);
    if (!link.source_name) errors.push(`Link #${i + 1}: missing source_name`);
    if (!link.sink_name) errors.push(`Link #${i + 1}: missing sink_name`);
  });

  return errors;
}
