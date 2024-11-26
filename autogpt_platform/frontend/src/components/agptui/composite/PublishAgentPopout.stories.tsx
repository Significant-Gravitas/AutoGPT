import type { Meta, StoryObj } from "@storybook/react";
import { PublishAgentPopout } from "@/components/agptui/composite/PublishAgentPopout";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPT UI/Composite/Publish Agent Popout",
  component: PublishAgentPopout,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    agents: { control: "object" },
    onOpenBuilder: { action: "onOpenBuilder" },
  },
} satisfies Meta<typeof PublishAgentPopout>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockAgents = [
  {
    name: "Marketing Assistant",
    lastEdited: "2 days ago",
    imageSrc: "https://picsum.photos/seed/marketing/300/200",
  },
  {
    name: "Sales Bot",
    lastEdited: "5 days ago", 
    imageSrc: "https://picsum.photos/seed/sales/300/200",
  },
  {
    name: "Content Writer",
    lastEdited: "1 week ago",
    imageSrc: "https://picsum.photos/seed/content/300/200",
  }
];

export const Default: Story = {
  args: {
    agents: mockAgents,
  },
};

export const WithCustomTrigger: Story = {
  args: {
    agents: mockAgents,
    trigger: <button>Custom Publish Button</button>,
  },
};

export const EmptyAgentsList: Story = {
  args: {
    agents: [],
  },
};

export const WithBuilderCallback: Story = {
  args: {
    agents: mockAgents,
    onOpenBuilder: () => {
      console.log("Opening builder...");
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const publishButton = canvas.getByText("Publish Agent");
    await userEvent.click(publishButton);
    
    const builderButton = canvas.getByText("Create new agent");
    await userEvent.click(builderButton);
  },
};

export const SelectAndPublishFlow: Story = {
  args: {
    agents: mockAgents,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    
    // Open popout
    const publishButton = canvas.getByText("Publish Agent");
    await userEvent.click(publishButton);
    
    // Select an agent
    const agentCard = canvas.getByText("Marketing Assistant");
    await userEvent.click(agentCard);
    
    // Click next
    const nextButton = canvas.getByText("Next");
    await userEvent.click(nextButton);
  },
};
