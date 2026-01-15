import { GraphInputSchema } from "@/lib/autogpt-server-api";
import { GraphMetaLike, IncompatibilityInfo } from "./types";

// Helper type for schema properties - the generated types are too loose
type SchemaProperties = Record<string, GraphInputSchema["properties"][string]>;
type SchemaRequired = string[];

// Helper to safely extract schema properties
export function getSchemaProperties(schema: unknown): SchemaProperties {
  if (
    schema &&
    typeof schema === "object" &&
    "properties" in schema &&
    typeof schema.properties === "object" &&
    schema.properties !== null
  ) {
    return schema.properties as SchemaProperties;
  }
  return {};
}

export function getSchemaRequired(schema: unknown): SchemaRequired {
  if (
    schema &&
    typeof schema === "object" &&
    "required" in schema &&
    Array.isArray(schema.required)
  ) {
    return schema.required as SchemaRequired;
  }
  return [];
}

/**
 * Creates the updated agent node inputs for a sub-agent node
 */
export function createUpdatedAgentNodeInputs(
  currentInputs: Record<string, unknown>,
  latestSubGraphVersion: GraphMetaLike,
): Record<string, unknown> {
  return {
    ...currentInputs,
    graph_version: latestSubGraphVersion.version,
    input_schema: latestSubGraphVersion.input_schema,
    output_schema: latestSubGraphVersion.output_schema,
  };
}

/** Generic edge type that works with both builders:
 *  - New builder uses CustomEdge with (formally) optional handles
 *  - Legacy builder uses ConnectedEdge type with required handles */
export type EdgeLike = {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
};

/**
 * Determines which edges are broken after an incompatible update.
 * Works with both legacy ConnectedEdge and new CustomEdge.
 */
export function getBrokenEdgeIDs(
  connections: EdgeLike[],
  incompatibilities: IncompatibilityInfo,
  nodeID: string,
): string[] {
  const brokenEdgeIDs: string[] = [];
  const typeMismatchInputNames = new Set(
    incompatibilities.inputTypeMismatches.map((m) => m.name),
  );

  connections.forEach((conn) => {
    // Check if this connection uses a missing input (node is target)
    if (
      conn.target === nodeID &&
      conn.targetHandle &&
      incompatibilities.missingInputs.includes(conn.targetHandle)
    ) {
      brokenEdgeIDs.push(conn.id);
    }

    // Check if this connection uses an input with a type mismatch (node is target)
    if (
      conn.target === nodeID &&
      conn.targetHandle &&
      typeMismatchInputNames.has(conn.targetHandle)
    ) {
      brokenEdgeIDs.push(conn.id);
    }

    // Check if this connection uses a missing output (node is source)
    if (
      conn.source === nodeID &&
      conn.sourceHandle &&
      incompatibilities.missingOutputs.includes(conn.sourceHandle)
    ) {
      brokenEdgeIDs.push(conn.id);
    }
  });

  return brokenEdgeIDs;
}
