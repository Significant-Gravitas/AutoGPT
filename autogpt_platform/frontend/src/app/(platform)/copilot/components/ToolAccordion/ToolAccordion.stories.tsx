import type { Meta, StoryObj } from "@storybook/nextjs";
import { GearIcon, LightningIcon } from "@phosphor-icons/react";
import { ToolAccordion } from "./ToolAccordion";
import { ContentMessage } from "./AccordionContent";

const meta: Meta<typeof ToolAccordion> = {
  title: "CoPilot/Tools/ToolAccordion",
  component: ToolAccordion,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Collapsible accordion used to wrap tool invocation results in the CoPilot chat.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="max-w-[480px]">
        <Story />
      </div>
    ),
  ],
};
export default meta;
type Story = StoryObj<typeof ToolAccordion>;

export const Collapsed: Story = {
  args: {
    icon: <GearIcon width={16} height={16} />,
    title: "Running GetWeather block",
    children: <ContentMessage>Block executed successfully.</ContentMessage>,
    defaultExpanded: false,
  },
};

export const Expanded: Story = {
  args: {
    icon: <LightningIcon width={16} height={16} />,
    title: "Agent execution started",
    children: (
      <ContentMessage>The agent is now processing your request.</ContentMessage>
    ),
    defaultExpanded: true,
  },
};

export const WithDescription: Story = {
  args: {
    icon: <GearIcon width={16} height={16} />,
    title: "FindBlocks",
    description: "Searching for blocks matching your query...",
    children: <ContentMessage>Found 3 matching blocks.</ContentMessage>,
    defaultExpanded: true,
  },
};
