import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { BlockCost } from "@/app/api/__generated__/models/blockCost";
import { BlockInfoCategoriesItem } from "@/app/api/__generated__/models/blockInfoCategoriesItem";
import { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import { NodeModelMetadata } from "@/app/api/__generated__/models/nodeModelMetadata";
import { preprocessInputSchema } from "@/components/renderers/InputRenderer/utils/input-schema-pre-processor";
import { cn } from "@/lib/utils";
import { RJSFSchema } from "@rjsf/utils";
import { NodeProps, Node as XYNode } from "@xyflow/react";
import React from "react";
import { BlockUIType } from "../../../types";
import { FormCreator } from "../FormCreator";
import { OutputHandler } from "../OutputHandler";
import { AyrshareConnectButton } from "./components/AyrshareConnectButton";
import { NodeAdvancedToggle } from "./components/NodeAdvancedToggle";
import { NodeContainer } from "./components/NodeContainer";
import { NodeExecutionBadge } from "./components/NodeExecutionBadge";
import { NodeHeader } from "./components/NodeHeader";
import { NodeDataRenderer } from "./components/NodeOutput/NodeOutput";
import { NodeRightClickMenu } from "./components/NodeRightClickMenu";
import { StickyNoteBlock } from "./components/StickyNoteBlock";
import { WebhookDisclaimer } from "./components/WebhookDisclaimer";
import { SubAgentUpdateFeature } from "./components/SubAgentUpdate/SubAgentUpdateFeature";
import { useCustomNode } from "./useCustomNode";

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
    const { inputSchema, outputSchema } = useCustomNode({ data, nodeId });

    const isAgent = data.uiType === BlockUIType.AGENT;

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

    const node = (
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
              "bg-white px-4",
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

    return (
      <NodeRightClickMenu
        nodeId={nodeId}
        subGraphID={data.hardcodedValues?.graph_id}
      >
        {node}
      </NodeRightClickMenu>
    );
  },
);

CustomNode.displayName = "CustomNode";
