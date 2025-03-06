import type { Meta, StoryObj } from "@storybook/react";
import { Agent, PublishAgentSelect } from "./PublishAgentSelect";

const meta: Meta<typeof PublishAgentSelect> = {
  title: "AGPT UI/Publish Agent Select",
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
    id: "1",
    version: 1,
  },
  {
    name: "Data Analyzer",
    lastEdited: "1 week ago",
    imageSrc: "https://picsum.photos/seed/data/300/200",
    id: "1",
    version: 1,
  },
  {
    name: "Image Recognition",
    lastEdited: "2 weeks ago",
    imageSrc: "https://picsum.photos/seed/image/300/200",
    id: "1",
    version: 1,
  },
  {
    name: "Chatbot Assistant",
    lastEdited: "3 weeks ago",
    imageSrc: "https://picsum.photos/seed/chat/300/200",
    id: "1",
    version: 1,
  },
  {
    name: "Code Generator",
    lastEdited: "1 month ago",
    imageSrc: "https://picsum.photos/seed/code/300/200",
    id: "1",
    version: 1,
  },
  {
    name: "AI Translator",
    lastEdited: "6 weeks ago",
    imageSrc: "https://picsum.photos/seed/translate/300/200",
    id: "1",
    version: 1,
  },
  {
    name: "Voice Assistant",
    lastEdited: "2 months ago",
    imageSrc: "https://picsum.photos/seed/voice/300/200",
    id: "1",
    version: 1,
  },
  {
    name: "Data Visualizer",
    lastEdited: "3 months ago",
    imageSrc: "https://picsum.photos/seed/visualize/300/200",
    id: "1",
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

export const SixAgents: Story = {
  args: {
    ...defaultArgs,
    agents: mockAgents.slice(0, 6),
  },
};

export const NineAgents: Story = {
  args: {
    ...defaultArgs,
    agents: mockAgents,
  },
};
