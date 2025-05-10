import type { Meta, StoryObj } from "@storybook/react";
import { Agent, PublishAgentSelect } from "./PublishAgentSelect";
import { userEvent, within, expect } from "@storybook/test";

const meta: Meta<typeof PublishAgentSelect> = {
  title: "Agpt Custom UI/marketing/Publish Agent Select",
  decorators: [
    (Story) => (
      <div className="backdrop-blur-4 flex h-screen items-center justify-center bg-black/40">
        <Story />
      </div>
    ),
  ],
  component: PublishAgentSelect,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof PublishAgentSelect>;

const mockAgents: Agent[] = [
  {
    name: "SEO Optimizer",
    lastEdited: "2 days ago",
    imageSrc: "https://picsum.photos/seed/seo/300/200",
    id: "1",
    version: 1,
  },
  {
    name: "Content Writer",
    lastEdited: "5 days ago",
    imageSrc: "https://picsum.photos/seed/writer/300/200",
    id: "2",
    version: 1,
  },
  {
    name: "Data Analyzer",
    lastEdited: "1 week ago",
    imageSrc: "https://picsum.photos/seed/data/300/200",
    id: "3",
    version: 1,
  },
  {
    name: "Image Recognition",
    lastEdited: "2 weeks ago",
    imageSrc: "https://picsum.photos/seed/image/300/200",
    id: "9",
    version: 1,
  },
  {
    name: "Chatbot Assistant",
    lastEdited: "3 weeks ago",
    imageSrc: "https://picsum.photos/seed/chat/300/200",
    id: "4",
    version: 1,
  },
  {
    name: "Code Generator",
    lastEdited: "1 month ago",
    imageSrc: "https://picsum.photos/seed/code/300/200",
    id: "5",
    version: 1,
  },
  {
    name: "AI Translator",
    lastEdited: "6 weeks ago",
    imageSrc: "https://picsum.photos/seed/translate/300/200",
    id: "6",
    version: 1,
  },
  {
    name: "Voice Assistant",
    lastEdited: "2 months ago",
    imageSrc: "https://picsum.photos/seed/voice/300/200",
    id: "7",
    version: 1,
  },
  {
    name: "Data Visualizer",
    lastEdited: "3 months ago",
    imageSrc: "https://picsum.photos/seed/visualize/300/200",
    id: "8",
    version: 1,
  },
];

const defaultArgs = {
  onSelect: (agentName: string) => console.log(`Selected: ${agentName}`),
  onCancel: () => console.log("Cancelled"),
  onNext: () => console.log("Next clicked"),
  onOpenBuilder: () => console.log("Open builder clicked"),
};

export const Default: Story = {
  args: {
    ...defaultArgs,
    agents: mockAgents,
  },
};

export const NoAgents: Story = {
  args: {
    ...defaultArgs,
    agents: [],
  },
};

export const SingleAgent: Story = {
  args: {
    ...defaultArgs,
    agents: [mockAgents[0]],
  },
};

export const TestingInteractions: Story = {
  args: {
    ...defaultArgs,
    agents: mockAgents,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // Select an agent
    const agentCard = canvas.getByText("SEO Optimizer");
    await userEvent.click(agentCard);

    // Click next button
    const nextButton = canvas.getByText(/next/i);
    await userEvent.click(nextButton);
  },
};
