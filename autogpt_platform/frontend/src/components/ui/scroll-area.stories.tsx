import React from "react";
import type { Meta, StoryObj } from "@storybook/react";

import { ScrollArea } from "./scroll-area";

const meta = {
  title: "UI/ScrollArea",
  component: ScrollArea,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    className: { control: "text" },
  },
} satisfies Meta<typeof ScrollArea>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    className: "h-[200px] w-[350px] rounded-md border p-4",
    children: (
      <div>
        <p className="mb-4">This is a scrollable area with some content.</p>
        {Array(20)
          .fill(0)
          .map((_, i) => (
            <div key={i} className="mb-2">
              Item {i + 1}
            </div>
          ))}
      </div>
    ),
  },
};

export const HorizontalScroll: Story = {
  args: {
    className: "h-[100px] w-[350px] rounded-md border",
    children: (
      <div className="flex p-4">
        {Array(20)
          .fill(0)
          .map((_, i) => (
            <div
              key={i}
              className="mr-4 flex h-16 w-16 items-center justify-center rounded-md border"
            >
              {i + 1}
            </div>
          ))}
      </div>
    ),
  },
};

export const NestedScrollAreas: Story = {
  args: {
    className: "h-[300px] w-[350px] rounded-md border p-4",
    children: (
      <div>
        <h4 className="mb-4 text-sm font-medium leading-none">
          Outer Scroll Area
        </h4>
        {Array(3)
          .fill(0)
          .map((_, i) => (
            <div key={i} className="mb-4">
              <p className="mb-2">Section {i + 1}</p>
              <ScrollArea className="h-[100px] w-[300px] rounded-md border p-4">
                <div>
                  <h5 className="mb-2 text-sm font-medium leading-none">
                    Inner Scroll Area
                  </h5>
                  {Array(10)
                    .fill(0)
                    .map((_, j) => (
                      <div key={j} className="mb-2">
                        Nested Item {j + 1}
                      </div>
                    ))}
                </div>
              </ScrollArea>
            </div>
          ))}
      </div>
    ),
  },
};

export const CustomScrollbarColors: Story = {
  args: {
    className: "h-[200px] w-[350px] rounded-md border p-4",
    children: (
      <div>
        <p className="mb-4">Customized scrollbar colors.</p>
        {Array(20)
          .fill(0)
          .map((_, i) => (
            <div key={i} className="mb-2">
              Item {i + 1}
            </div>
          ))}
      </div>
    ),
  },
  parameters: {
    backgrounds: { default: "dark" },
  },
  decorators: [
    (Story) => (
      <div className="dark">
        <style>
          {`
            .dark .custom-scrollbar [data-radix-scroll-area-thumb] {
              background-color: rgba(255, 255, 255, 0.2);
            }
          `}
        </style>
        <Story />
      </div>
    ),
  ],
};
