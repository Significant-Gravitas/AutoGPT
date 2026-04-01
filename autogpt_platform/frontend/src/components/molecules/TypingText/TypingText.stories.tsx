import { TypingText } from "./TypingText";
import type { Meta, StoryObj } from "@storybook/nextjs";

const meta: Meta<typeof TypingText> = {
  title: "Molecules/TypingText",
  tags: ["autodocs"],
  component: TypingText,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Animates text appearing character by character with a blinking cursor. Useful for loading states and onboarding flows.",
      },
    },
  },
  argTypes: {
    text: { control: "text" },
    active: { control: "boolean" },
    speed: { control: { type: "range", min: 10, max: 100, step: 5 } },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    text: "Personalizing your experience...",
    active: true,
    speed: 30,
  },
};

export const Fast: Story = {
  args: {
    text: "This types out quickly!",
    active: true,
    speed: 15,
  },
};

export const Slow: Story = {
  args: {
    text: "This types out slowly...",
    active: true,
    speed: 80,
  },
};

export const Inactive: Story = {
  args: {
    text: "This text is hidden because active is false",
    active: false,
  },
};
