import type { Meta, StoryObj } from "@storybook/nextjs";
import type { BlockOutputResponse } from "@/app/api/__generated__/models/blockOutputResponse";
import { BlockOutputCard } from "./BlockOutputCard";

const meta: Meta<typeof BlockOutputCard> = {
  title: "CoPilot/Tools/RunBlock/BlockOutputCard",
  component: BlockOutputCard,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Renders the output of a successful block execution, grouped by output key.",
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
type Story = StoryObj<typeof BlockOutputCard>;

export const SingleOutput: Story = {
  args: {
    output: {
      block_id: "block-1",
      block_name: "GetWeather",
      message: "Block executed successfully.",
      outputs: {
        temperature: [22.5],
      },
      success: true,
    } as BlockOutputResponse,
  },
};

export const MultipleOutputs: Story = {
  args: {
    output: {
      block_id: "block-2",
      block_name: "SearchWeb",
      message: "Found 3 results.",
      outputs: {
        title: ["Result 1", "Result 2", "Result 3"],
        url: [
          "https://example.com/1",
          "https://example.com/2",
          "https://example.com/3",
        ],
        snippet: [
          "First result snippet...",
          "Second result snippet...",
          "Third result snippet...",
        ],
      },
      success: true,
    } as BlockOutputResponse,
  },
};
