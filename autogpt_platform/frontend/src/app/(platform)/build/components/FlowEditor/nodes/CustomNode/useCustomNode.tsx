import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { CustomNodeData } from "./CustomNode";
import { BlockUIType } from "../../../types";
import { useMemo } from "react";
import { mergeSchemaForResolution } from "./helpers";
import { SpecialBlockID } from "@/lib/autogpt-server-api";

/**
 * Build a dynamic input schema for MCP blocks.
 *
 * When a tool has been selected (tool_input_schema is populated), the block
 * should render:
 *   1. The credentials field (from the static schema)
 *   2. The selected tool's input parameters (from tool_input_schema)
 *
 * Static fields like server_url, selected_tool, available_tools, and
 * tool_arguments are hidden because they're pre-configured from the dialog.
 */
function buildMCPInputSchema(
  staticSchema: Record<string, any>,
  toolInputSchema: Record<string, any>,
): Record<string, any> {
  const credentialsProp = staticSchema.properties?.credentials;
  const staticRequired = staticSchema.required ?? [];

  return {
    type: "object",
    properties: {
      ...(credentialsProp ? { credentials: credentialsProp } : {}),
      ...(toolInputSchema.properties ?? {}),
    },
    required: [
      ...staticRequired.filter((r: string) => r === "credentials"),
      ...(toolInputSchema.required ?? []),
    ],
  };
}

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
  const isMCPWithTool =
    data.block_id === SpecialBlockID.MCP_TOOL &&
    !!data.hardcodedValues?.tool_input_schema?.properties;

  const currentInputSchema = isAgent
    ? (data.hardcodedValues.input_schema ?? {})
    : isMCPWithTool
      ? buildMCPInputSchema(
          data.inputSchema,
          data.hardcodedValues.tool_input_schema,
        )
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
    isMCPWithTool,
  };
};
