import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { CustomNodeData } from "./CustomNode";
import { BlockUIType } from "../../../types";
import { useMemo } from "react";
import { mergeSchemaForResolution } from "./helpers";

export const useCustomNode = ({
  data,
  nodeId,
}: {
  data: CustomNodeData;
  nodeId: string;
}) => {
  const isInResolutionMode = useNodeStore((state) =>
    state.nodesInResolutionMode.has(nodeId),
  );
  const resolutionData = useNodeStore((state) =>
    state.nodeResolutionData.get(nodeId),
  );

  const isAgent = data.uiType === BlockUIType.AGENT;

  const currentInputSchema = isAgent
    ? (data.hardcodedValues.input_schema ?? {})
    : data.inputSchema;
  const currentOutputSchema = isAgent
    ? (data.hardcodedValues.output_schema ?? {})
    : data.outputSchema;

  const inputSchema = useMemo(() => {
    if (isAgent && isInResolutionMode && resolutionData) {
      return mergeSchemaForResolution(
        resolutionData.currentSchema.input_schema,
        resolutionData.pendingUpdate.input_schema,
        resolutionData,
        "input",
      );
    }
    return currentInputSchema;
  }, [isAgent, isInResolutionMode, resolutionData, currentInputSchema]);

  const outputSchema = useMemo(() => {
    if (isAgent && isInResolutionMode && resolutionData) {
      return mergeSchemaForResolution(
        resolutionData.currentSchema.output_schema,
        resolutionData.pendingUpdate.output_schema,
        resolutionData,
        "output",
      );
    }
    return currentOutputSchema;
  }, [isAgent, isInResolutionMode, resolutionData, currentOutputSchema]);

  return {
    inputSchema,
    outputSchema,
  };
};
