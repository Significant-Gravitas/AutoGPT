import type { Meta, StoryObj } from "@storybook/react";

import {
  Toast,
  ToastProvider,
  ToastViewport,
  ToastTitle,
  ToastDescription,
  ToastClose,
  ToastAction,
} from "./toast";

const meta = {
  title: "UI/Toast",
  component: Toast,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    variant: {
      control: "select",
      options: ["default", "destructive"],
    },
  },
  decorators: [
    (Story) => (
      <ToastProvider>
        <Story />
        <ToastViewport />
      </ToastProvider>
    ),
  ],
} satisfies Meta<typeof Toast>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: (
      <>
        <ToastTitle>Default Toast</ToastTitle>
        <ToastDescription>This is a default toast message.</ToastDescription>
      </>
    ),
  },
};

export const Destructive: Story = {
  args: {
    variant: "destructive",
    children: (
      <>
        <ToastTitle>Destructive Toast</ToastTitle>
        <ToastDescription>
          This is a destructive toast message.
        </ToastDescription>
      </>
    ),
  },
};

export const WithAction: Story = {
  args: {
    children: (
      <>
        <ToastTitle>Toast with Action</ToastTitle>
        <ToastDescription>This toast has an action button.</ToastDescription>
        <ToastAction altText="Try again">Try again</ToastAction>
      </>
    ),
  },
};

export const WithClose: Story = {
  args: {
    children: (
      <>
        <ToastTitle>Closable Toast</ToastTitle>
        <ToastDescription>This toast can be closed.</ToastDescription>
        <ToastClose />
      </>
    ),
  },
};

export const CustomContent: Story = {
  args: {
    children: (
      <>
        <div className="flex items-center">
          <span className="mr-2">🎉</span>
          <ToastTitle>Custom Toast</ToastTitle>
        </div>
        <ToastDescription>This toast has custom content.</ToastDescription>
      </>
    ),
  },
};
