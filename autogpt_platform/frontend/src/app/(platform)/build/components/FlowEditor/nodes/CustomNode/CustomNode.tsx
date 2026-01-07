import React, { useMemo } from "react";
import { Node as XYNode, NodeProps } from "@xyflow/react";
import { RJSFSchema } from "@rjsf/utils";
import { BlockUIType } from "../../../types";
import { StickyNoteBlock } from "./components/StickyNoteBlock";
import { BlockInfoCategoriesItem } from "@/app/api/__generated__/models/blockInfoCategoriesItem";
import { BlockCost } from "@/app/api/__generated__/models/blockCost";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import { NodeContainer } from "./components/NodeContainer";
import { NodeHeader } from "./components/NodeHeader";
import { FormCreator } from "../FormCreator";
import { preprocessInputSchema } from "@/components/renderers/input-renderer/utils/input-schema-pre-processor";
import { OutputHandler } from "../OutputHandler";
import { NodeAdvancedToggle } from "./components/NodeAdvancedToggle";
import { NodeDataRenderer } from "./components/NodeOutput/NodeOutput";
import { NodeExecutionBadge } from "./components/NodeExecutionBadge";
import { cn } from "@/lib/utils";
import { WebhookDisclaimer } from "./components/WebhookDisclaimer";
import { AyrshareConnectButton } from "./components/AyrshareConnectButton";
import { NodeModelMetadata } from "@/app/api/__generated__/models/nodeModelMetadata";
import { SubAgentUpdateFeature } from "./components/SubAgentUpdate/SubAgentUpdateFeature";
import { useNodeStore, NodeResolutionData } from "../../../../stores/nodeStore";

type SchemaProperties = Record<string, unknown>;

/**
 * Merges schemas during resolution mode to include removed inputs/outputs
 * that still have connections, so users can see and delete them.
 */
function mergeSchemaForResolution(
  currentSchema: Record<string, unknown>,
  newSchema: Record<string, unknown>,
  resolutionData: NodeResolutionData,
  type: "input" | "output",
): Record<string, unknown> {
  const newProps = (newSchema.properties as SchemaProperties) || {};
  const currentProps = (currentSchema.properties as SchemaProperties) || {};
  const mergedProps = { ...newProps };
  const incomp = resolutionData.incompatibilities;

  if (type === "input") {
    // Add back missing inputs that have connections
    incomp.missingInputs.forEach((inputName: string) => {
      if (currentProps[inputName]) {
        mergedProps[inputName] = currentProps[inputName];
      }
    });
    // Add back inputs with type mismatches (keep old type so connection works visually)
    incomp.inputTypeMismatches.forEach(
      (mismatch: { name: string; oldType: string; newType: string }) => {
        if (currentProps[mismatch.name]) {
          mergedProps[mismatch.name] = currentProps[mismatch.name];
        }
      },
    );
  } else {
    // Add back missing outputs that have connections
    incomp.missingOutputs.forEach((outputName: string) => {
      if (currentProps[outputName]) {
        mergedProps[outputName] = currentProps[outputName];
      }
    });
  }

  return {
    ...newSchema,
    properties: mergedProps,
  };
}

export type CustomNodeData = {
  hardcodedValues: {
    [key: string]: any;
  };
  title: string;
  description: string;
  inputSchema: RJSFSchema;
  outputSchema: RJSFSchema;
  uiType: BlockUIType;
  block_id: string;
  status?: AgentExecutionStatus;
  nodeExecutionResult?: NodeExecutionResult;
  staticOutput?: boolean;
  // TODO : We need better type safety for the following backend fields.
  costs: BlockCost[];
  categories: BlockInfoCategoriesItem[];
  metadata?: NodeModelMetadata;
  errors?: { [key: string]: string };
};

export type CustomNode = XYNode<CustomNodeData, "custom">;

export const CustomNode: React.FC<NodeProps<CustomNode>> = React.memo(
  ({ data, id: nodeId, selected }) => {
    // Subscribe to the actual resolution mode state for this node
    const isInResolutionMode = useNodeStore((state) =>
      state.nodesInResolutionMode.has(nodeId),
    );
    const resolutionData = useNodeStore((state) =>
      state.nodeResolutionData.get(nodeId),
    );

    const isAgent = data.uiType === BlockUIType.AGENT;

    // Get base schemas for agent nodes
    const currentInputSchema = isAgent
      ? (data.hardcodedValues.input_schema ?? {})
      : data.inputSchema;
    const currentOutputSchema = isAgent
      ? (data.hardcodedValues.output_schema ?? {})
      : data.outputSchema;

    // During resolution mode, merge old connected inputs/outputs with new schema
    // so users can see and delete the broken connections
    const inputSchema = useMemo(() => {
      if (isAgent && isInResolutionMode && resolutionData) {
        // Use the stored old schema from resolution data for merging
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
        // Use the stored old schema from resolution data for merging
        return mergeSchemaForResolution(
          resolutionData.currentSchema.output_schema,
          resolutionData.pendingUpdate.output_schema,
          resolutionData,
          "output",
        );
      }
      return currentOutputSchema;
    }, [isAgent, isInResolutionMode, resolutionData, currentOutputSchema]);

    // Handle sticky note separately
    if (data.uiType === BlockUIType.NOTE) {
      return (
        <StickyNoteBlock data={data} selected={selected} nodeId={nodeId} />
      );
    }

    const showHandles =
      data.uiType !== BlockUIType.INPUT &&
      data.uiType !== BlockUIType.WEBHOOK &&
      data.uiType !== BlockUIType.WEBHOOK_MANUAL;

    const isWebhook = [
      BlockUIType.WEBHOOK,
      BlockUIType.WEBHOOK_MANUAL,
    ].includes(data.uiType);

    const isAyrshare = data.uiType === BlockUIType.AYRSHARE;

    const hasConfigErrors =
      data.errors &&
      Object.values(data.errors).some(
        (value) => value !== null && value !== undefined && value !== "",
      );

    const outputData = data.nodeExecutionResult?.output_data;
    const hasOutputError =
      typeof outputData === "object" &&
      outputData !== null &&
      "error" in outputData;

    const hasErrors = hasConfigErrors || hasOutputError;

    // Currently all blockTypes design are similar - that's why i am using the same component for all of them
    // If in future - if we need some drastic change in some blockTypes design - we can create separate components for them
    return (
      <NodeContainer selected={selected} nodeId={nodeId} hasErrors={hasErrors}>
        <div className="rounded-xlarge bg-white">
          <NodeHeader data={data} nodeId={nodeId} />
          {isAgent && <SubAgentUpdateFeature nodeID={nodeId} nodeData={data} />}
          {isWebhook && <WebhookDisclaimer nodeId={nodeId} />}
          {isAyrshare && <AyrshareConnectButton />}
          <FormCreator
            jsonSchema={preprocessInputSchema(inputSchema)}
            nodeId={nodeId}
            uiType={data.uiType}
            className={cn(
              "my-4 bg-white pr-6",
              isWebhook && "pointer-events-none opacity-50",
            )}
            showHandles={showHandles}
          />
          <NodeAdvancedToggle nodeId={nodeId} />
          {data.uiType != BlockUIType.OUTPUT && (
            <OutputHandler
              uiType={data.uiType}
              outputSchema={outputSchema}
              nodeId={nodeId}
            />
          )}
          <NodeDataRenderer nodeId={nodeId} />
        </div>
        <NodeExecutionBadge nodeId={nodeId} />
      </NodeContainer>
    );
  },
);

CustomNode.displayName = "CustomNode";
