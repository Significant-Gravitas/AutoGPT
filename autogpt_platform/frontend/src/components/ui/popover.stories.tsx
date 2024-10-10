import React from "react";
import type { Meta, StoryObj } from "@storybook/react";

import { Popover, PopoverTrigger, PopoverContent } from "./popover";
import { Button } from "./button";

const meta = {
  title: "UI/Popover",
  component: Popover,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof Popover>;

export default meta;
type Story = StoryObj<typeof meta>;

const PopoverExample = (args: any) => (
  <Popover>
    <PopoverTrigger asChild>
      <Button variant="outline">Open Popover</Button>
    </PopoverTrigger>
    <PopoverContent>
      <div className="text-sm">
        <h3 className="mb-1 font-medium">Popover Content</h3>
        <p>This is the content of the popover.</p>
      </div>
    </PopoverContent>
  </Popover>
);

export const Default: Story = {
  render: () => <PopoverExample />,
};

export const AlignStart: Story = {
  render: () => (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline">Open Popover</Button>
      </PopoverTrigger>
      <PopoverContent align="start">
        <div className="text-sm">
          <h3 className="mb-1 font-medium">Popover Content</h3>
          <p>This is the content of the popover.</p>
        </div>
      </PopoverContent>
    </Popover>
  ),
};

export const AlignEnd: Story = {
  render: () => (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline">Open Popover</Button>
      </PopoverTrigger>
      <PopoverContent align="end">
        <div className="text-sm">
          <h3 className="mb-1 font-medium">Popover Content</h3>
          <p>This is the content of the popover.</p>
        </div>
      </PopoverContent>
    </Popover>
  ),
};

export const CustomOffset: Story = {
  render: () => (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline">Open Popover</Button>
      </PopoverTrigger>
      <PopoverContent sideOffset={10}>
        <div className="text-sm">
          <h3 className="mb-1 font-medium">Popover Content</h3>
          <p>This is the content of the popover.</p>
        </div>
      </PopoverContent>
    </Popover>
  ),
};

export const CustomContent: Story = {
  render: () => (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline">Custom Popover</Button>
      </PopoverTrigger>
      <PopoverContent>
        <div className="text-sm">
          <h3 className="mb-1 font-medium">Custom Content</h3>
          <p>This popover has custom content.</p>
          <Button className="mt-2" size="sm">
            Action Button
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  ),
};
