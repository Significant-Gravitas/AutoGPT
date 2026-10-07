import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { TimePicker } from "./TimePicker";

const meta: Meta<typeof TimePicker> = {
  title: "Molecules/TimePicker",
  component: TimePicker,
  tags: ["autodocs"],
  parameters: {
    a11y: { test: "error" },
    layout: "centered",
    docs: {
      description: {
        component:
          "Compact time selector using three small Selects (hour, minute, AM/PM).",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  render: function BasicStory() {
    const [value, setValue] = useState<string>("12:00");
    return <TimePicker value={value} onChange={setValue} />;
  },
};
