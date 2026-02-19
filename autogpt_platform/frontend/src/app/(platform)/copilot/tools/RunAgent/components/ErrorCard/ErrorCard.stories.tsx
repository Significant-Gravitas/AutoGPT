import type { Meta, StoryObj } from "@storybook/nextjs";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import { ErrorCard } from "./ErrorCard";

const meta: Meta<typeof ErrorCard> = {
  title: "CoPilot/Tools/RunAgent/ErrorCard",
  component: ErrorCard,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Displays an error response from an agent execution attempt.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div style={{ maxWidth: 480 }}>
        <Story />
      </div>
    ),
  ],
};
export default meta;
type Story = StoryObj<typeof ErrorCard>;

export const SimpleError: Story = {
  args: {
    output: {
      message: "Agent not found in the store.",
    } as ErrorResponse,
  },
};

export const WithDetails: Story = {
  args: {
    output: {
      message: "Agent execution failed.",
      error: "RuntimeError: Graph execution timed out.",
      details: JSON.stringify(
        { graph_id: "graph-xyz", timeout_ms: 30000 },
        null,
        2,
      ),
    } as ErrorResponse,
  },
};
