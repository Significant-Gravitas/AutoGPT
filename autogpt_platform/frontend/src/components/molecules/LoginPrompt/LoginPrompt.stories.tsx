import type { Meta, StoryObj } from "@storybook/nextjs";
import { LoginPrompt } from "./LoginPrompt";

const meta = {
  title: "Molecules/LoginPrompt",
  component: LoginPrompt,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
  args: {
    onLogin: () => console.log("Login clicked"),
    onContinueAsGuest: () => console.log("Continue as guest clicked"),
  },
} satisfies Meta<typeof LoginPrompt>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    message: "Please log in to save your chat history and access your account",
  },
};

export const CustomMessage: Story = {
  args: {
    message:
      "To continue with this agent and save your progress, please sign in to your account",
  },
};

export const ShortMessage: Story = {
  args: {
    message: "Sign in to continue",
  },
};
