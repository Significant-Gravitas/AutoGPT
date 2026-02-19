import type { Meta, StoryObj } from "@storybook/nextjs";
import type { AgentDetailsResponse } from "@/app/api/__generated__/models/agentDetailsResponse";
import { withCopilotChatActions } from "../../../../components/_storybook/CopilotChatActionsDecorator";
import { AgentDetailsCard } from "./AgentDetailsCard";

const meta: Meta<typeof AgentDetailsCard> = {
  title: "CoPilot/Tools/RunAgent/AgentDetailsCard",
  component: AgentDetailsCard,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Shows agent details with options to run with example values or custom inputs.",
      },
    },
  },
  decorators: [
    withCopilotChatActions,
    (Story) => (
      <div className="max-w-[480px]">
        <Story />
      </div>
    ),
  ],
};
export default meta;
type Story = StoryObj<typeof AgentDetailsCard>;

export const Default: Story = {
  args: {
    output: {
      message: "Here are the details for the WeatherReporter agent.",
      agent: {
        id: "agent-1",
        name: "WeatherReporter",
        description: "Fetches and reports weather for any city.",
      },
      user_authenticated: true,
    } as AgentDetailsResponse,
  },
};

export const WithInputSchema: Story = {
  args: {
    output: {
      message: "Agent found. You can run it with your own inputs.",
      agent: {
        id: "agent-2",
        name: "EmailDrafter",
        description: "Drafts professional emails based on your instructions.",
        inputs: {
          type: "object",
          properties: {
            recipient: { type: "string", title: "Recipient" },
            subject: { type: "string", title: "Subject" },
            tone: { type: "string", title: "Tone", default: "professional" },
          },
          required: ["recipient", "subject"],
        },
      },
      user_authenticated: true,
    } as AgentDetailsResponse,
  },
};
