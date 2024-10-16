import React from "react";
import type { Meta, StoryObj } from "@storybook/react";

import { ContentRenderer } from "./render";

const meta = {
  title: "UI/ContentRenderer",
  component: ContentRenderer,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    value: { control: "text" },
    truncateLongData: { control: "boolean" },
  },
} satisfies Meta<typeof ContentRenderer>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Text: Story = {
  args: {
    value: "This is a simple text content.",
  },
};

export const LongText: Story = {
  args: {
    value:
      "This is a very long text that will be truncated when the truncateLongData prop is set to true. It contains more than 100 characters to demonstrate the truncation feature.",
    truncateLongData: true,
  },
};

export const Image: Story = {
  args: {
    value: "https://example.com/image.jpg",
  },
};

export const Video: Story = {
  args: {
    value: "https://example.com/video.mp4",
  },
};

export const YouTubeVideo: Story = {
  args: {
    value: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  },
};

export const JsonObject: Story = {
  args: {
    value: { key: "value", nested: { array: [1, 2, 3] } },
  },
};

export const TruncatedJsonObject: Story = {
  args: {
    value: {
      key: "value",
      nested: { array: [1, 2, 3] },
      longText:
        "This is a very long text that will be truncated when rendered as part of the JSON object.",
    },
    truncateLongData: true,
  },
};
