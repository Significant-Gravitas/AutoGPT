import type { Meta, StoryObj } from "@storybook/nextjs";
import { ExecutionStartedMessage } from "./ExecutionStartedMessage";

const meta = {
  title: "Molecules/ExecutionStartedMessage",
  component: ExecutionStartedMessage,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof ExecutionStartedMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    executionId: "exec-123e4567-e89b-12d3-a456-426614174000",
    agentName: "Data Analysis Agent",
    onViewExecution: () => console.log("View execution clicked"),
  },
};

export const WithoutAgentName: Story = {
  args: {
    executionId: "exec-987f6543-a21b-45c6-b789-123456789abc",
    onViewExecution: () => console.log("View execution clicked"),
  },
};

export const CustomMessage: Story = {
  args: {
    executionId: "exec-456a7890-b12c-34d5-e678-901234567def",
    agentName: "Email Automation Agent",
    message: "Your email automation agent is now processing emails",
    onViewExecution: () => console.log("View execution clicked"),
  },
};

export const WithoutViewButton: Story = {
  args: {
    executionId: "exec-789b1234-c56d-78e9-f012-345678901abc",
    agentName: "Social Media Manager",
  },
};

export const LongAgentName: Story = {
  args: {
    executionId: "exec-321d5678-e90f-12a3-b456-789012345cde",
    agentName:
      "Advanced Multi-Platform Social Media Content Manager and Analytics Agent",
    message:
      "Your advanced automation agent has started processing multiple tasks in the background",
    onViewExecution: () => console.log("View execution clicked"),
  },
};

export const ShortExecutionId: Story = {
  args: {
    executionId: "exec-123",
    agentName: "Quick Agent",
    onViewExecution: () => console.log("View execution clicked"),
  },
};
