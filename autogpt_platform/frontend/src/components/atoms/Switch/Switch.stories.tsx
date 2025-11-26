import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { Switch } from "./Switch";

const meta: Meta<typeof Switch> = {
  title: "Atoms/Switch",
  component: Switch,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Shadcn-based toggle switch. Controlled via checked and onCheckedChange.",
      },
    },
  },
  argTypes: {
    checked: { control: "boolean", description: "Checked state (controlled)." },
    disabled: { control: "boolean", description: "Disable the switch." },
    onCheckedChange: { action: "change", description: "Change handler." },
    className: { control: "text", description: "Optional className." },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  render: function BasicStory(args) {
    const [on, setOn] = useState<boolean>(true);
    return (
      <div className="flex items-center gap-3">
        <Switch
          aria-label="Toggle"
          checked={on}
          onCheckedChange={(v) => {
            setOn(v);
            if (args.onCheckedChange) args.onCheckedChange(v);
          }}
        />
        <span className="text-sm">{on ? "On" : "Off"}</span>
      </div>
    );
  },
};

export const Disabled: Story = {
  args: { disabled: true },
  render: function DisabledStory(args) {
    return <Switch aria-label="Disabled switch" disabled {...args} />;
  },
};

export const WithLabel: Story = {
  render: function WithLabelStory() {
    const [on, setOn] = useState<boolean>(false);
    const id = "ds-switch-label";
    return (
      <label htmlFor={id} className="flex items-center gap-3">
        <Switch id={id} checked={on} onCheckedChange={setOn} />
        <span className="text-sm">Enable notifications</span>
      </label>
    );
  },
};
