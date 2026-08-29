import type { Meta, StoryObj } from "@storybook/nextjs";
import { LLMItem } from "./LLMItem";

const meta: Meta<typeof LLMItem> = {
  title: "Atoms/LLMItem",
  tags: ["autodocs"],
  component: LLMItem,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "LLMItem component for displaying different LLM providers with their respective icons and names.",
      },
    },
  },
  argTypes: {
    type: {
      control: "select",
      options: ["claude", "gpt", "perplexity"],
      description: "LLM provider type",
    },
  },
  args: {
    type: "claude",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Claude: Story = {
  args: {
    type: "claude",
  },
};

export const GPT: Story = {
  args: {
    type: "gpt",
  },
};

export const Perplexity: Story = {
  args: {
    type: "perplexity",
  },
};

export const AllVariants: Story = {
  render: renderAllVariants,
};

function renderAllVariants() {
  return (
    <div className="flex flex-wrap gap-8">
      <LLMItem type="claude" />
      <LLMItem type="gpt" />
      <LLMItem type="perplexity" />
    </div>
  );
}
