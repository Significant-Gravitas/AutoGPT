import type { Meta, StoryObj } from "@storybook/nextjs";
import { AgentCarouselMessage } from "./AgentCarouselMessage";

const meta = {
  title: "Molecules/AgentCarouselMessage",
  component: AgentCarouselMessage,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof AgentCarouselMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

const sampleAgents = [
  {
    id: "agent-1",
    name: "Data Analysis Agent",
    description:
      "Analyzes CSV and Excel files, generates insights and visualizations",
    version: 1,
  },
  {
    id: "agent-2",
    name: "Web Scraper",
    description:
      "Extracts data from websites and formats it into structured JSON",
    version: 2,
  },
  {
    id: "agent-3",
    name: "Email Automation",
    description:
      "Automates email responses based on custom rules and templates",
    version: 1,
  },
  {
    id: "agent-4",
    name: "Social Media Manager",
    description:
      "Schedules and publishes posts across multiple social media platforms",
    version: 3,
  },
];

export const SingleAgent: Story = {
  args: {
    agents: [sampleAgents[0]],
    onSelectAgent: (id) => console.log("Selected agent:", id),
  },
};

export const TwoAgents: Story = {
  args: {
    agents: sampleAgents.slice(0, 2),
    onSelectAgent: (id) => console.log("Selected agent:", id),
  },
};

export const FourAgents: Story = {
  args: {
    agents: sampleAgents,
    onSelectAgent: (id) => console.log("Selected agent:", id),
  },
};

export const ManyAgentsWithTotal: Story = {
  args: {
    agents: sampleAgents,
    totalCount: 15,
    onSelectAgent: (id) => console.log("Selected agent:", id),
  },
};

export const WithoutVersion: Story = {
  args: {
    agents: [
      {
        id: "agent-1",
        name: "Basic Agent",
        description: "A simple agent without version information",
      },
      {
        id: "agent-2",
        name: "Another Agent",
        description: "Another agent without version",
      },
    ],
    onSelectAgent: (id) => console.log("Selected agent:", id),
  },
};

export const LongDescriptions: Story = {
  args: {
    agents: [
      {
        id: "agent-1",
        name: "Complex Agent",
        description:
          "This agent performs multiple complex tasks including data analysis, report generation, automated email responses, integration with third-party APIs, and much more. It's designed to handle large-scale operations efficiently.",
        version: 1,
      },
      {
        id: "agent-2",
        name: "Advanced Automation Agent",
        description:
          "An advanced automation solution that connects to various services, processes data in real-time, sends notifications, generates reports, and maintains comprehensive logs of all operations performed.",
        version: 2,
      },
    ],
    onSelectAgent: (id) => console.log("Selected agent:", id),
  },
};

export const WithoutSelectHandler: Story = {
  args: {
    agents: sampleAgents.slice(0, 2),
  },
};
