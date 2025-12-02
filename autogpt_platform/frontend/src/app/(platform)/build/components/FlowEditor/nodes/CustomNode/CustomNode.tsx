import React from "react";
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
};

export type CustomNode = XYNode<CustomNodeData, "custom">;

export const CustomNode: React.FC<NodeProps<CustomNode>> = React.memo(
  ({ data, id: nodeId, selected }) => {
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

    const inputSchema =
      data.uiType === BlockUIType.AGENT
        ? (data.hardcodedValues.input_schema ?? {})
        : data.inputSchema;

    const outputSchema =
      data.uiType === BlockUIType.AGENT
        ? (data.hardcodedValues.output_schema ?? {})
        : data.outputSchema;

    // Currently all blockTypes design are similar - that's why i am using the same component for all of them
    // If in future - if we need some drastic change in some blockTypes design - we can create separate components for them
    return (
      <NodeContainer selected={selected} nodeId={nodeId}>
        <div className="rounded-xlarge bg-white">
          <NodeHeader data={data} nodeId={nodeId} />
          {isWebhook && <WebhookDisclaimer nodeId={nodeId} />}
          {isAyrshare && <AyrshareConnectButton />}
          <FormCreator
            jsonSchema={preprocessInputSchema(inputSchema)}
            nodeId={nodeId}
            uiType={data.uiType}
            className={cn(
              "bg-white pr-6",
              isWebhook && "pointer-events-none opacity-50",
            )}
            showHandles={showHandles}
          />
          <NodeAdvancedToggle nodeId={nodeId} />
          {data.uiType != BlockUIType.OUTPUT && (
            <OutputHandler outputSchema={outputSchema} nodeId={nodeId} />
          )}
          <NodeDataRenderer nodeId={nodeId} />
        </div>
        <NodeExecutionBadge nodeId={nodeId} />
      </NodeContainer>
    );
  },
);

CustomNode.displayName = "CustomNode";
