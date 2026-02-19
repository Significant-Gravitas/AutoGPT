import type { Meta, StoryObj } from "@storybook/nextjs";
import type { ExecutionStartedResponse } from "@/app/api/__generated__/models/executionStartedResponse";
import { ExecutionStartedCard } from "./ExecutionStartedCard";

const meta: Meta<typeof ExecutionStartedCard> = {
  title: "CoPilot/Tools/RunAgent/ExecutionStartedCard",
  component: ExecutionStartedCard,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Confirmation card shown when an agent execution has been successfully started.",
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
type Story = StoryObj<typeof ExecutionStartedCard>;

export const Default: Story = {
  args: {
    output: {
      execution_id: "exec-abc-123",
      graph_id: "graph-xyz-456",
      graph_name: "WeatherReporter",
      message: "Agent execution started successfully.",
    } as ExecutionStartedResponse,
  },
};

export const WithLibraryLink: Story = {
  args: {
    output: {
      execution_id: "exec-def-789",
      graph_id: "graph-uvw-012",
      graph_name: "EmailSender",
      message: "Agent execution started. You can monitor progress below.",
      library_agent_link: "/library/agents/agent-id-123",
    } as ExecutionStartedResponse,
  },
};
