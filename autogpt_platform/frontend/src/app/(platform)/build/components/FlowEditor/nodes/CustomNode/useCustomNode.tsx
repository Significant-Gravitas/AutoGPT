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
 * renders only the selected tool's input parameters. Credentials are NOT
 * included because authentication is already handled by the MCP dialog's
 * OAuth flow and stored server-side.
 *
 * Static fields like server_url, selected_tool, available_tools, and
 * tool_arguments are hidden because they're pre-configured from the dialog.
 */
function buildMCPInputSchema(
  toolInputSchema: Record<string, any>,
): Record<string, any> {
  return {
    type: "object",
    properties: {
      ...(toolInputSchema.properties ?? {}),
    },
    required: [...(toolInputSchema.required ?? [])],
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
      ? buildMCPInputSchema(data.hardcodedValues.tool_input_schema)
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
