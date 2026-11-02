import { useControlPanelStore } from "@/app/(platform)/build/stores/controlPanelStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { ReactFlowProvider } from "@xyflow/react";
import { NuqsAdapter } from "nuqs/adapters/react";
import type { CustomNode } from "../FlowEditor/nodes/CustomNode/CustomNode";
import { BlockUIType } from "../types";
import { NewControlPanel } from "./NewControlPanel";

function makeNode(
  id: string,
  title: string,
  description: string,
  inputs: string[],
  outputs: string[],
  customizedName?: string,
): CustomNode {
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues: {},
      title,
      description,
      inputSchema: {
        type: "object",
        properties: Object.fromEntries(inputs.map((n) => [n, {}])),
      },
      outputSchema: {
        type: "object",
        properties: Object.fromEntries(outputs.map((n) => [n, {}])),
      },
      uiType: BlockUIType.STANDARD,
      block_id: `block-${id}`,
      costs: [],
      categories: [],
      metadata: customizedName ? { customized_name: customizedName } : {},
    },
  };
}

const NODES: CustomNode[] = [
  makeNode("1", "AgentInputBlock", "Collects the topic", ["name"], ["result"]),
  makeNode(
    "2",
    "AITextGeneratorBlock",
    "Writes the summary",
    ["prompt", "model"],
    ["response"],
    "Morning summary",
  ),
  makeNode(
    "3",
    "SendEmailBlock",
    "Emails the summary",
    ["to_email", "subject", "body"],
    ["status"],
  ),
];

const meta = {
  title: "Build/NewControlPanel",
  component: NewControlPanel,
  parameters: { layout: "fullscreen" },
  beforeEach() {
    useControlPanelStore.getState().reset();
    useNodeStore.setState({ nodes: NODES });
    return () => {
      useControlPanelStore.getState().reset();
      useNodeStore.setState({ nodes: [] });
    };
  },
  decorators: [
    (Story) => (
      <BackendAPIProvider>
        <NuqsAdapter>
          <ReactFlowProvider>
            <TooltipProvider>
              <div className="relative h-[calc(100vh-4rem)] min-h-[32rem] w-full rounded-medium bg-surface-hover">
                <Story />
              </div>
            </TooltipProvider>
          </ReactFlowProvider>
        </NuqsAdapter>
      </BackendAPIProvider>
    ),
  ],
} satisfies Meta<typeof NewControlPanel>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Editable workflow: blocks, search, save and undo/redo, split by the three
 *  separators the panel draws. */
export const Editable: Story = {};

/** Read-only workflow: only the search button, so no separators. */
export const ReadOnly: Story = {
  args: { isReadOnly: true },
};

/** The search popover open from the panel, listing the workflow's nodes. */
export const GraphSearchOpen: Story = {
  beforeEach() {
    useControlPanelStore.getState().setGraphSearchOpen(true);
  },
};
