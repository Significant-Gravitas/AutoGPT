import type { Meta, StoryObj } from "@storybook/nextjs";
import { TypeWriter } from "./Typewriter";

const meta: Meta<typeof TypeWriter> = {
  title: "Molecules/Typewriter",
  component: TypeWriter,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Reveals text character by character with a blinking cursor effect.",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  args: {
    text: "Hello, world!",
  },
};

export const LongText: Story = {
  args: {
    text: "This is a longer piece of text that demonstrates how the typewriter effect handles multiple words and sentences.",
  },
};

export const WithCustomClassName: Story = {
  args: {
    text: "Styled text",
    className: "text-2xl font-bold text-blue-500",
  },
};
