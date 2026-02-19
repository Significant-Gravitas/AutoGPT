import type { Meta, StoryObj } from "@storybook/nextjs";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import { ErrorCard } from "./ErrorCard";

const meta: Meta<typeof ErrorCard> = {
  title: "CoPilot/Tools/RunBlock/ErrorCard",
  component: ErrorCard,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Displays an error response from a block execution, including optional error and details fields.",
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
      message: "Block execution failed: timeout after 30 seconds.",
    } as ErrorResponse,
  },
};

export const WithDetails: Story = {
  args: {
    output: {
      message: "Block execution failed.",
      details: JSON.stringify(
        { block_id: "abc-123", retry_count: 3, last_attempt: "2026-02-19" },
        null,
        2,
      ),
    } as ErrorResponse,
  },
};

export const WithErrorAndDetails: Story = {
  args: {
    output: {
      message: "Authentication failed for GetWeather block.",
      error: "InvalidCredentialsError: API key is expired or invalid.",
      details: JSON.stringify(
        { provider: "openweather", status_code: 401 },
        null,
        2,
      ),
    } as ErrorResponse,
  },
};
