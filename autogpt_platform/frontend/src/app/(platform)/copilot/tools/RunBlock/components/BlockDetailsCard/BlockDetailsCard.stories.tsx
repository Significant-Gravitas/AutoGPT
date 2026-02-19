import type { Meta, StoryObj } from "@storybook/nextjs";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { BlockDetailsResponse } from "../../helpers";
import { BlockDetailsCard } from "./BlockDetailsCard";

const meta: Meta<typeof BlockDetailsCard> = {
  title: "CoPilot/Tools/RunBlock/BlockDetailsCard",
  component: BlockDetailsCard,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="max-w-[480px]">
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

const baseBlock: BlockDetailsResponse = {
  type: ResponseType.block_details,
  message:
    "Here are the details for the GetWeather block. Provide the required inputs to run it.",
  session_id: "session-123",
  user_authenticated: true,
  block: {
    id: "block-abc-123",
    name: "GetWeather",
    description: "Fetches current weather data for a given location.",
    inputs: {
      type: "object",
      properties: {
        location: {
          title: "Location",
          type: "string",
          description:
            "City name or coordinates (e.g. 'London' or '51.5,-0.1')",
        },
        units: {
          title: "Units",
          type: "string",
          description: "Temperature units: 'metric' or 'imperial'",
        },
      },
      required: ["location"],
    },
    outputs: {
      type: "object",
      properties: {
        temperature: {
          title: "Temperature",
          type: "number",
          description: "Current temperature in the requested units",
        },
        condition: {
          title: "Condition",
          type: "string",
          description: "Weather condition description (e.g. 'Sunny', 'Rain')",
        },
      },
    },
    credentials: [],
  },
};

export const Default: Story = {
  args: {
    output: baseBlock,
  },
};

export const InputsOnly: Story = {
  args: {
    output: {
      ...baseBlock,
      message: "This block requires inputs. No outputs are defined.",
      block: {
        ...baseBlock.block,
        outputs: {},
      },
    },
  },
};

export const OutputsOnly: Story = {
  args: {
    output: {
      ...baseBlock,
      message: "This block has no required inputs.",
      block: {
        ...baseBlock.block,
        inputs: {},
      },
    },
  },
};

export const ManyFields: Story = {
  args: {
    output: {
      ...baseBlock,
      message: "Block with many input and output fields.",
      block: {
        ...baseBlock.block,
        name: "SendEmail",
        description: "Sends an email via SMTP.",
        inputs: {
          type: "object",
          properties: {
            to: {
              title: "To",
              type: "string",
              description: "Recipient email address",
            },
            subject: {
              title: "Subject",
              type: "string",
              description: "Email subject line",
            },
            body: {
              title: "Body",
              type: "string",
              description: "Email body content",
            },
            cc: {
              title: "CC",
              type: "string",
              description: "CC recipients (comma-separated)",
            },
            bcc: {
              title: "BCC",
              type: "string",
              description: "BCC recipients (comma-separated)",
            },
          },
          required: ["to", "subject", "body"],
        },
        outputs: {
          type: "object",
          properties: {
            message_id: {
              title: "Message ID",
              type: "string",
              description: "Unique ID of the sent email",
            },
            status: {
              title: "Status",
              type: "string",
              description: "Delivery status",
            },
          },
        },
      },
    },
  },
};

export const NoFieldDescriptions: Story = {
  args: {
    output: {
      ...baseBlock,
      message: "Fields without descriptions.",
      block: {
        ...baseBlock.block,
        name: "SimpleBlock",
        inputs: {
          type: "object",
          properties: {
            input_a: { title: "Input A", type: "string" },
            input_b: { title: "Input B", type: "number" },
          },
          required: ["input_a"],
        },
        outputs: {
          type: "object",
          properties: {
            result: { title: "Result", type: "string" },
          },
        },
      },
    },
  },
};
