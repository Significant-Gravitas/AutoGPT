import type { Meta, StoryObj } from "@storybook/nextjs";
import { MorphingTextAnimation } from "./MorphingTextAnimation";

const meta: Meta<typeof MorphingTextAnimation> = {
  title: "CoPilot/Loaders/MorphingTextAnimation",
  component: MorphingTextAnimation,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Animated text that morphs in letter by letter using framer-motion.",
      },
    },
  },
};
export default meta;
type Story = StoryObj<typeof MorphingTextAnimation>;

export const Short: Story = {
  args: { text: "Loading..." },
};

export const Long: Story = {
  args: { text: "Analyzing your request and preparing the best response" },
};

export const Emoji: Story = {
  args: { text: "🚀 Building your agent..." },
};
