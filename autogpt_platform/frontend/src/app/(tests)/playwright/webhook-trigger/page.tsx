"use client";

import { ReactFlow, ReactFlowProvider } from "@xyflow/react";
import {
  CustomNode,
  type CustomNode as FlowNode,
  type CustomNodeData,
} from "@/app/(platform)/build/components/legacy-builder/CustomNode/CustomNode";
import { BuilderContext } from "@/app/(platform)/build/components/legacy-builder/Flow/Flow";
import {
  BlockUIType,
  type Category,
  type Node as GraphNode,
} from "@/lib/autogpt-server-api";
import { useMemo } from "react";

const mockCategories: Category[] = [
  {
    category: "triggers",
    description: "Trigger block",
  },
];

const mockWebhook: NonNullable<GraphNode["webhook"]> = {
  id: "webhook-manual-123",
  url: "https://hooks.autogpt.io/very/long/path/that/should/wrap/properly/when/rendered/in/the/manual/webhook/block/and-demonstrate-that-break-all-is-working/callback?id=1234567890abcdefghijklmnopqrstuvwxyz",
  provider: "http",
  credentials_id: "",
  webhook_type: "manual",
  resource: "generic",
  events: ["POST"],
  secret: "****",
  config: {},
};

const mockNodeData: CustomNodeData = {
  blockType: "WebhookManualBlock",
  blockCosts: [],
  title: "Generic Webhook (Manual)",
  description: "Trigger flows manually via incoming POST requests.",
  categories: mockCategories,
  inputSchema: {
    type: "object",
    properties: {},
    required: [],
  },
  outputSchema: {
    type: "object",
    properties: {},
    required: [],
  },
  hardcodedValues: {},
  connections: [],
  webhook: mockWebhook,
  isOutputOpen: false,
  uiType: BlockUIType.WEBHOOK_MANUAL,
  block_id: "manual-webhook-block",
  executionResults: [],
  errors: {},
  metadata: {},
};

const mockBuilderContext = {
  libraryAgent: null,
  visualizeBeads: "no" as const,
  setIsAnyModalOpen: () => undefined,
  getNextNodeId: () => "mock-node-id",
  getNodeTitle: () => mockNodeData.title,
};

export default function WebhookTriggerPreviewPage() {
  const nodeData = useMemo(
    () => ({
      ...mockNodeData,
      inputSchema: { ...mockNodeData.inputSchema },
      outputSchema: { ...mockNodeData.outputSchema },
      hardcodedValues: { ...mockNodeData.hardcodedValues },
      connections: [...mockNodeData.connections],
      categories: mockNodeData.categories.map((category) => ({ ...category })),
      executionResults: mockNodeData.executionResults
        ? [...mockNodeData.executionResults]
        : mockNodeData.executionResults,
      webhook: mockNodeData.webhook
        ? { ...mockNodeData.webhook }
        : mockNodeData.webhook,
    }),
    [],
  );

  const nodes = useMemo<FlowNode[]>(
    () => [
      {
        id: "manual-webhook-node",
        type: "custom",
        position: { x: 0, y: 0 },
        data: nodeData,
      },
    ],
    [nodeData],
  );

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100 p-8">
      <ReactFlowProvider>
        <BuilderContext.Provider value={mockBuilderContext}>
          <div className="h-[600px] w-[640px] rounded-2xl bg-white p-6 shadow-lg">
            <ReactFlow
              nodes={nodes}
              edges={[]}
              nodeTypes={{ custom: CustomNode }}
              panOnScroll={false}
              zoomOnScroll={false}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable={false}
              fitView
              fitViewOptions={{ padding: 0.2 }}
            />
          </div>
        </BuilderContext.Provider>
      </ReactFlowProvider>
    </div>
  );
}
