import type { Meta, StoryObj } from "@storybook/nextjs";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { withCopilotChatActions } from "../../../../components/_storybook/CopilotChatActionsDecorator";
import { SetupRequirementsCard } from "./SetupRequirementsCard";

const meta: Meta<typeof SetupRequirementsCard> = {
  title: "CoPilot/Tools/RunAgent/SetupRequirementsCard",
  component: SetupRequirementsCard,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Displays setup requirements for running an agent, including missing credentials and expected inputs.",
      },
    },
  },
  decorators: [
    withCopilotChatActions,
    (Story) => (
      <div style={{ maxWidth: 480 }}>
        <Story />
      </div>
    ),
  ],
};
export default meta;
type Story = StoryObj<typeof SetupRequirementsCard>;

export const WithCredentials: Story = {
  args: {
    output: {
      message: "This agent requires credentials before it can run.",
      setup_info: {
        agent_id: "agent-1",
        agent_name: "WeatherReporter",
        user_readiness: {
          missing_credentials: [
            {
              id: "openweather_key",
              title: "OpenWeather API Key",
              provider: "openweather",
              type: "api_key",
              required: true,
            },
          ],
        },
      },
    } as SetupRequirementsResponse,
  },
};

export const WithInputs: Story = {
  args: {
    output: {
      message: "This agent needs the following inputs to run.",
      setup_info: {
        agent_id: "agent-2",
        agent_name: "EmailDrafter",
        requirements: {
          inputs: [
            {
              name: "recipient",
              title: "Recipient",
              type: "string",
              required: true,
              description: "Email address of the recipient",
            },
            {
              name: "subject",
              title: "Subject",
              type: "string",
              required: true,
            },
            {
              name: "body_hint",
              title: "Body Hint",
              type: "string",
              required: false,
              description: "Optional hint for email body",
            },
          ],
        },
      },
    } as SetupRequirementsResponse,
  },
};
