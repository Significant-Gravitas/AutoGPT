import type { Meta, StoryObj } from "@storybook/nextjs";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { withCopilotChatActions } from "../../../../components/_storybook/CopilotChatActionsDecorator";
import { SetupRequirementsCard } from "./SetupRequirementsCard";

const meta: Meta<typeof SetupRequirementsCard> = {
  title: "CoPilot/Tools/RunBlock/SetupRequirementsCard",
  component: SetupRequirementsCard,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Displays setup requirements for running a block, including missing credentials and expected inputs.",
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

export const WithInputs: Story = {
  args: {
    output: {
      message:
        "This block requires some inputs before it can run. Please provide the values below.",
      setup_info: {
        agent_id: "block-1",
        agent_name: "GetWeather",
        requirements: {
          inputs: [
            {
              name: "location",
              title: "Location",
              type: "string",
              required: true,
              description: "City name or coordinates",
            },
            {
              name: "units",
              title: "Units",
              type: "string",
              required: false,
              description: "metric or imperial",
            },
          ],
        },
      },
    } as SetupRequirementsResponse,
  },
};

export const WithCredentials: Story = {
  args: {
    output: {
      message: "This block requires credentials to be configured.",
      setup_info: {
        agent_id: "block-2",
        agent_name: "SendEmail",
        user_readiness: {
          missing_credentials: [
            {
              id: "smtp_cred",
              title: "SMTP Credentials",
              provider: "smtp",
              type: "api_key",
              required: true,
            },
          ],
        },
      },
    } as SetupRequirementsResponse,
  },
};
