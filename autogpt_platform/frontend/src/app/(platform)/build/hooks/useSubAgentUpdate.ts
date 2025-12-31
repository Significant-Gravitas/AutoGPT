import { useMemo } from "react";
import {
  GraphMeta as LegacyGraphMeta,
  GraphInputSchema,
  GraphOutputSchema,
} from "@/lib/autogpt-server-api";
import { GraphMeta as GeneratedGraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { getEffectiveType } from "@/lib/utils";
import { ConnectionData } from "../components/legacy-builder/CustomNode/CustomNode";

// Union type for GraphMeta that works with both legacy and new builder
export type GraphMetaLike = LegacyGraphMeta | GeneratedGraphMeta;

// Generic edge type that works with both builders
// Legacy builder uses ConnectionData with `edge_id`, new builder uses CustomEdge with `id`
export type EdgeLike = {
  id?: string;
  edge_id?: string;
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
};

// Helper type for schema properties - the generated types are too loose
type SchemaProperties = Record<string, GraphInputSchema["properties"][string]>;
type SchemaRequired = string[];

// Helper to safely extract schema properties
function getSchemaProperties(schema: unknown): SchemaProperties {
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

function getSchemaRequired(schema: unknown): SchemaRequired {
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

export type IncompatibilityInfo = {
  missingInputs: string[]; // Connected inputs that no longer exist
  missingOutputs: string[]; // Connected outputs that no longer exist
  newInputs: string[]; // Inputs that exist in new version but not in current
  newOutputs: string[]; // Outputs that exist in new version but not in current
  newRequiredInputs: string[]; // New required inputs not in current version or not required
  inputTypeMismatches: Array<{
    name: string;
    oldType: string;
    newType: string;
  }>; // Connected inputs where the type has changed
};

export type SubAgentUpdateInfo<T extends GraphMetaLike = GraphMetaLike> = {
  hasUpdate: boolean;
  currentVersion: number;
  latestVersion: number;
  latestFlow: T | null;
  isCompatible: boolean;
  incompatibilities: IncompatibilityInfo | null;
};

/**
 * Checks if a newer version of a sub-agent is available and determines compatibility
 */
export function useSubAgentUpdate<T extends GraphMetaLike>(
  nodeId: string,
  graphID: string | undefined,
  graphVersion: number | undefined,
  currentInputSchema: GraphInputSchema | undefined,
  currentOutputSchema: GraphOutputSchema | undefined,
  connections: ConnectionData | EdgeLike[],
  availableFlows: T[],
): SubAgentUpdateInfo<T> {
  // Find the latest version of the same graph
  const latestFlow = useMemo(() => {
    if (!graphID) return null;
    return availableFlows.find((flow) => flow.id === graphID) || null;
  }, [graphID, availableFlows]);

  // Check if there's an update available
  const hasUpdate = useMemo(() => {
    if (!latestFlow || graphVersion === undefined) return false;
    return latestFlow.version! > graphVersion;
  }, [latestFlow, graphVersion]);

  // Get connected input and output handles for this specific node
  const connectedHandles = useMemo(() => {
    const inputHandles = new Set<string>();
    const outputHandles = new Set<string>();

    connections.forEach((conn) => {
      // If this node is the target, the targetHandle is an input on this node
      if (conn.target === nodeId && conn.targetHandle) {
        inputHandles.add(conn.targetHandle);
      }
      // If this node is the source, the sourceHandle is an output on this node
      if (conn.source === nodeId && conn.sourceHandle) {
        outputHandles.add(conn.sourceHandle);
      }
    });

    return { inputHandles, outputHandles };
  }, [connections, nodeId]);

  // Check schema compatibility
  const compatibilityResult = useMemo((): {
    isCompatible: boolean;
    incompatibilities: IncompatibilityInfo | null;
  } => {
    if (!hasUpdate || !latestFlow) {
      return { isCompatible: true, incompatibilities: null };
    }

    const newInputProps = getSchemaProperties(latestFlow.input_schema);
    const newOutputProps = getSchemaProperties(latestFlow.output_schema);
    const newRequiredInputs = getSchemaRequired(latestFlow.input_schema);

    const currentInputProps = getSchemaProperties(currentInputSchema);
    const currentOutputProps = getSchemaProperties(currentOutputSchema);
    const currentRequiredInputs = getSchemaRequired(currentInputSchema);

    const incompatibilities: IncompatibilityInfo = {
      missingInputs: [],
      missingOutputs: [],
      newInputs: [],
      newOutputs: [],
      newRequiredInputs: [],
      inputTypeMismatches: [],
    };

    // Check for missing connected inputs and type mismatches
    connectedHandles.inputHandles.forEach((inputHandle) => {
      if (!(inputHandle in newInputProps)) {
        incompatibilities.missingInputs.push(inputHandle);
      } else {
        // Check for type mismatch on connected inputs
        const currentProp = currentInputProps[inputHandle];
        const newProp = newInputProps[inputHandle];
        const currentType = getEffectiveType(currentProp);
        const newType = getEffectiveType(newProp);

        if (currentType && newType && currentType !== newType) {
          incompatibilities.inputTypeMismatches.push({
            name: inputHandle,
            oldType: currentType,
            newType: newType,
          });
        }
      }
    });

    // Check for missing connected outputs
    connectedHandles.outputHandles.forEach((outputHandle) => {
      if (!(outputHandle in newOutputProps)) {
        incompatibilities.missingOutputs.push(outputHandle);
      }
    });

    // Check for new required inputs that didn't exist or weren't required before
    newRequiredInputs.forEach((requiredInput) => {
      const existedBefore = requiredInput in currentInputProps;
      const wasRequiredBefore = currentRequiredInputs.includes(
        requiredInput as string,
      );

      if (!existedBefore || !wasRequiredBefore) {
        incompatibilities.newRequiredInputs.push(requiredInput as string);
      }
    });

    // Check for new inputs that don't exist in the current version
    Object.keys(newInputProps).forEach((inputName) => {
      if (!(inputName in currentInputProps)) {
        incompatibilities.newInputs.push(inputName);
      }
    });

    // Check for new outputs that don't exist in the current version
    Object.keys(newOutputProps).forEach((outputName) => {
      if (!(outputName in currentOutputProps)) {
        incompatibilities.newOutputs.push(outputName);
      }
    });

    const hasIncompatibilities =
      incompatibilities.missingInputs.length > 0 ||
      incompatibilities.missingOutputs.length > 0 ||
      incompatibilities.newRequiredInputs.length > 0 ||
      incompatibilities.inputTypeMismatches.length > 0;

    return {
      isCompatible: !hasIncompatibilities,
      incompatibilities: hasIncompatibilities ? incompatibilities : null,
    };
  }, [
    hasUpdate,
    latestFlow,
    currentInputSchema,
    currentOutputSchema,
    connectedHandles,
  ]);

  return {
    hasUpdate,
    currentVersion: graphVersion || 0,
    latestVersion: latestFlow?.version || 0,
    latestFlow,
    isCompatible: compatibilityResult.isCompatible,
    incompatibilities: compatibilityResult.incompatibilities,
  };
}

/**
 * Creates the updated hardcoded values for a sub-agent node
 */
export function createUpdatedHardcodedValues(
  currentHardcodedValues: Record<string, unknown>,
  latestFlow: GraphMetaLike,
): Record<string, unknown> {
  return {
    ...currentHardcodedValues,
    graph_version: latestFlow.version,
    input_schema: latestFlow.input_schema,
    output_schema: latestFlow.output_schema,
  };
}

/**
 * Determines which edges are broken after an incompatible update.
 * Works with both legacy ConnectionData (edge_id) and new CustomEdge (id).
 */
export function getBrokenEdgeIDs(
  connections: ConnectionData | EdgeLike[],
  incompatibilities: IncompatibilityInfo,
  nodeID: string,
): string[] {
  const brokenEdgeIds: string[] = [];
  const typeMismatchInputNames = new Set(
    incompatibilities.inputTypeMismatches.map((m) => m.name),
  );

  connections.forEach((conn) => {
    // Get edge ID - new builder uses `id`, legacy uses `edge_id`
    const edgeID = "id" in conn ? conn.id : conn.edge_id;
    if (!edgeID) return;

    // Check if this connection uses a missing input (node is target)
    if (
      conn.target === nodeID &&
      conn.targetHandle &&
      incompatibilities.missingInputs.includes(conn.targetHandle)
    ) {
      brokenEdgeIds.push(edgeID);
    }

    // Check if this connection uses an input with a type mismatch (node is target)
    if (
      conn.target === nodeID &&
      conn.targetHandle &&
      typeMismatchInputNames.has(conn.targetHandle)
    ) {
      brokenEdgeIds.push(edgeID);
    }

    // Check if this connection uses a missing output (node is source)
    if (
      conn.source === nodeID &&
      conn.sourceHandle &&
      incompatibilities.missingOutputs.includes(conn.sourceHandle)
    ) {
      brokenEdgeIds.push(edgeID);
    }
  });

  return brokenEdgeIds;
}
