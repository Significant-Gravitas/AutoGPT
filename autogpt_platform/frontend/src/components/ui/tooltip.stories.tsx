import React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { Button } from "./button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./tooltip";

const meta = {
  title: "UI/Tooltip",
  component: Tooltip,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
  argTypes: {
    children: { control: "text" },
    delayDuration: { control: "number" },
  },
} satisfies Meta<typeof Tooltip>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: (
      <>
        <TooltipTrigger asChild>
          <Button variant="outline">Hover me</Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>This is a tooltip</p>
        </TooltipContent>
      </>
    ),
  },
};

export const LongContent: Story = {
  args: {
    children: (
      <>
        <TooltipTrigger asChild>
          <Button variant="outline">Hover for long content</Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            This is a tooltip with longer content that might wrap to multiple
            lines.
          </p>
        </TooltipContent>
      </>
    ),
  },
};

export const CustomDelay: Story = {
  args: {
    delayDuration: 1000,
    children: (
      <>
        <TooltipTrigger asChild>
          <Button variant="outline">Hover with delay</Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>This tooltip has a 1 second delay</p>
        </TooltipContent>
      </>
    ),
  },
};

export const CustomContent: Story = {
  args: {
    children: (
      <>
        <TooltipTrigger asChild>
          <Button variant="outline">Hover for custom content</Button>
        </TooltipTrigger>
        <TooltipContent>
          <div className="flex items-center">
            <span className="mr-2">🚀</span>
            <p>Custom tooltip content</p>
          </div>
        </TooltipContent>
      </>
    ),
  },
};
