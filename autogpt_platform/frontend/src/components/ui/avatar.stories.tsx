import type { Meta, StoryObj } from "@storybook/react";

import { Avatar, AvatarImage, AvatarFallback } from "./avatar";

const meta = {
  title: "UI/Avatar",
  component: Avatar,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    // Add any specific controls for Avatar props here if needed
  },
} satisfies Meta<typeof Avatar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: <AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />,
  },
};

export const WithFallback: Story = {
  args: {
    children: (
      <>
        <AvatarImage src="/broken-image.jpg" alt="@shadcn" />
        <AvatarFallback>CN</AvatarFallback>
      </>
    ),
  },
};

export const FallbackOnly: Story = {
  args: {
    children: <AvatarFallback>JD</AvatarFallback>,
  },
};

export const CustomSize: Story = {
  args: {
    className: "h-16 w-16",
    children: <AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />,
  },
};

export const CustomContent: Story = {
  args: {
    children: (
      <AvatarFallback>
        <span role="img" aria-label="Rocket">
          🚀
        </span>
      </AvatarFallback>
    ),
  },
};
