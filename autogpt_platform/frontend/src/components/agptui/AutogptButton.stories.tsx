import type { Meta, StoryObj } from "@storybook/react";
import AutogptButton from "./AutogptButton";

const meta = {
  title: "Agpt UI/general/AutogptButton",
  component: AutogptButton,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof AutogptButton>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    children: "Primary Button",
    variant: "default",
  },
};

export const PrimaryWithIcon: Story = {
  args: {
    icon: true,
    children: "Primary Button",
    variant: "default",
  },
};

export const PrimaryDisabled: Story = {
  args: {
    children: "Primary Button",
    variant: "default",
    isDisabled: true,
  },
};

export const Secondary: Story = {
  args: {
    children: "Primary Button",
    variant: "secondary",
  },
};

export const SecondaryWithIcon: Story = {
  args: {
    icon: true,
    children: "Secondary Button",
    variant: "secondary",
  },
};

export const SecondaryDisabled: Story = {
  args: {
    children: "Secondary Button",
    variant: "secondary",
    isDisabled: true,
  },
};

export const Destructive: Story = {
  args: {
    children: "Destructive Button",
    variant: "destructive",
  },
};

export const DestructiveWithIcon: Story = {
  args: {
    icon: true,
    children: "Destructive Button",
    variant: "destructive",
  },
};

export const DestructiveDisabled: Story = {
  args: {
    children: "Destructive Button",
    variant: "destructive",
    isDisabled: true,
  },
};

export const Outline: Story = {
  args: {
    children: "Outline Button",
    variant: "outline",
  },
};

export const OutlineWithIcon: Story = {
  args: {
    icon: true,
    children: "Outline Button",
    variant: "outline",
  },
};

export const OutlineDisabled: Story = {
  args: {
    children: "Outline Button",
    variant: "outline",
    isDisabled: true,
  },
};

export const Ghost: Story = {
  args: {
    children: "Ghost Button",
    variant: "ghost",
  },
};

export const GhostDisabled: Story = {
  args: {
    children: "Ghost Button",
    variant: "ghost",
    isDisabled: true,
  },
};

export const Link: Story = {
  args: {
    children: "Link Button",
    variant: "link",
  },
};

export const Loading: Story = {
  args: {
    children: "Loading Button",
    isLoading: true,
  },
};
