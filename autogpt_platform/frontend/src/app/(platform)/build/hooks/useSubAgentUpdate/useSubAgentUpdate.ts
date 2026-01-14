import { useMemo } from "react";
import { GraphInputSchema, GraphOutputSchema } from "@/lib/autogpt-server-api";
import { getEffectiveType } from "@/lib/utils";
import { EdgeLike, getSchemaProperties, getSchemaRequired } from "./helpers";
import {
  GraphMetaLike,
  IncompatibilityInfo,
  SubAgentUpdateInfo,
} from "./types";

/**
 * Checks if a newer version of a sub-agent is available and determines compatibility
 */
export function useSubAgentUpdate<T extends GraphMetaLike>(
  nodeID: string,
  graphID: string | undefined,
  graphVersion: number | undefined,
  currentInputSchema: GraphInputSchema | undefined,
  currentOutputSchema: GraphOutputSchema | undefined,
  connections: EdgeLike[],
  availableGraphs: T[],
): SubAgentUpdateInfo<T> {
  // Find the latest version of the same graph
  const latestGraph = useMemo(() => {
    if (!graphID) return null;
    return availableGraphs.find((graph) => graph.id === graphID) || null;
  }, [graphID, availableGraphs]);

  // Check if there's an update available
  const hasUpdate = useMemo(() => {
    if (!latestGraph || graphVersion === undefined) return false;
    return latestGraph.version! > graphVersion;
  }, [latestGraph, graphVersion]);

  // Get connected input and output handles for this specific node
  const connectedHandles = useMemo(() => {
    const inputHandles = new Set<string>();
    const outputHandles = new Set<string>();

    connections.forEach((conn) => {
      // If this node is the target, the targetHandle is an input on this node
      if (conn.target === nodeID && conn.targetHandle) {
        inputHandles.add(conn.targetHandle);
      }
      // If this node is the source, the sourceHandle is an output on this node
      if (conn.source === nodeID && conn.sourceHandle) {
        outputHandles.add(conn.sourceHandle);
      }
    });

    return { inputHandles, outputHandles };
  }, [connections, nodeID]);

  // Check schema compatibility
  const compatibilityResult = useMemo((): {
    isCompatible: boolean;
    incompatibilities: IncompatibilityInfo | null;
  } => {
    if (!hasUpdate || !latestGraph) {
      return { isCompatible: true, incompatibilities: null };
    }

    const newInputProps = getSchemaProperties(latestGraph.input_schema);
    const newOutputProps = getSchemaProperties(latestGraph.output_schema);
    const newRequiredInputs = getSchemaRequired(latestGraph.input_schema);

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
    latestGraph,
    currentInputSchema,
    currentOutputSchema,
    connectedHandles,
  ]);

  return {
    hasUpdate,
    currentVersion: graphVersion || 0,
    latestVersion: latestGraph?.version || 0,
    latestGraph,
    isCompatible: compatibilityResult.isCompatible,
    incompatibilities: compatibilityResult.incompatibilities,
  };
}
