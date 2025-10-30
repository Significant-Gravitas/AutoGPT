import type { Meta, StoryObj } from "@storybook/nextjs";
import { QuickActionsWelcome } from "./QuickActionsWelcome";

const meta = {
  title: "Molecules/QuickActionsWelcome",
  component: QuickActionsWelcome,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
  args: {
    onActionClick: (action: string) => console.log("Action clicked:", action),
  },
} satisfies Meta<typeof QuickActionsWelcome>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    title: "Welcome to Agent Chat",
    description:
      "I can help you discover and run AI agents. Try one of these quick actions to get started:",
    actions: [
      "Find agents for data analysis",
      "Search for automation tools",
      "Show me popular agents",
      "Help me build a workflow",
    ],
  },
};

export const TwoActions: Story = {
  args: {
    title: "Get Started",
    description: "Choose an action to begin your journey with AI agents:",
    actions: ["Explore the marketplace", "Create a new agent"],
  },
};

export const SixActions: Story = {
  args: {
    title: "What would you like to do?",
    description: "Select from these options to get started:",
    actions: [
      "Search for agents",
      "Browse categories",
      "View my agents",
      "Create new workflow",
      "Import from template",
      "Learn about agents",
    ],
  },
};

export const Disabled: Story = {
  args: {
    title: "Welcome to Agent Chat",
    description:
      "I can help you discover and run AI agents. Try one of these quick actions to get started:",
    actions: [
      "Find agents for data analysis",
      "Search for automation tools",
      "Show me popular agents",
      "Help me build a workflow",
    ],
    disabled: true,
  },
};

export const LongTexts: Story = {
  args: {
    title: "Welcome to the AutoGPT Platform Agent Discovery System",
    description:
      "This interactive chat interface allows you to explore, discover, and execute AI-powered agents that can help you automate tasks, analyze data, and solve complex problems. Select one of the suggested actions below to begin your exploration journey:",
    actions: [
      "Find agents that can help me with advanced data analysis and visualization tasks",
      "Search for workflow automation agents that integrate with popular services",
      "Show me the most popular and highly-rated agents in the marketplace",
      "Help me understand how to build and deploy my own custom agent workflows",
    ],
  },
};

export const ShortTitle: Story = {
  args: {
    title: "Chat",
    description: "What would you like to do today?",
    actions: ["Find agents", "Browse marketplace", "View history", "Get help"],
  },
};
