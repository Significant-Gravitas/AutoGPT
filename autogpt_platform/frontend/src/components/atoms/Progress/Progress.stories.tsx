import type { Meta, StoryObj } from "@storybook/nextjs";
import { useEffect, useState } from "react";
import { Progress } from "./Progress";

const meta: Meta<typeof Progress> = {
  title: "Atoms/Progress",
  component: Progress,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Simple progress bar with value and optional max (default 100).",
      },
    },
  },
  argTypes: {
    value: {
      control: { type: "number", min: 0, max: 100 },
      description: "Current value.",
    },
    max: {
      control: { type: "number", min: 1 },
      description: "Maximum value (default 100).",
    },
    className: {
      control: "text",
      description: "Optional className for container (e.g. height).",
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  args: { value: 50 },
  render: function BasicStory(args) {
    return (
      <div className="w-80">
        <Progress {...args} />
      </div>
    );
  },
};

export const CustomMax: Story = {
  args: { value: 30, max: 60 },
  render: function CustomMaxStory(args) {
    return (
      <div className="w-80">
        <Progress {...args} />
      </div>
    );
  },
  parameters: {
    docs: {
      description: { story: "With max=60, value=30 renders as 50%." },
    },
  },
};

export const Sizes: Story = {
  render: function SizesStory() {
    return (
      <div className="w-80 space-y-4">
        <Progress value={40} className="h-1" />
        <Progress value={60} className="h-2" />
        <Progress value={80} className="h-3" />
      </div>
    );
  },
  parameters: {
    docs: {
      description: {
        story: "Adjust height via className (e.g., h-1, h-2, h-3).",
      },
    },
  },
};

export const Live: Story = {
  render: function LiveStory() {
    const [value, setValue] = useState<number>(0);
    useEffect(() => {
      const id = setInterval(
        () => setValue((v) => (v >= 100 ? 0 : v + 10)),
        400,
      );
      return () => clearInterval(id);
    }, []);
    return <Progress value={value} className="w-80" />;
  },
  parameters: {
    docs: {
      description: { story: "Animated example updating value on an interval." },
    },
  },
};
