import type { Meta, StoryObj } from "@storybook/nextjs";
import { Checkbox } from "./Checkbox";

const meta: Meta<typeof Checkbox> = {
  title: "Atoms/Checkbox",
  component: Checkbox,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Radix checkbox. Controlled via checked and onCheckedChange, or uncontrolled via defaultChecked.",
      },
    },
  },
  argTypes: {
    checked: { control: "boolean", description: "Checked state (controlled)." },
    disabled: { control: "boolean", description: "Disable the checkbox." },
    onCheckedChange: { action: "change", description: "Change handler." },
    className: { control: "text", description: "Optional className." },
  },
  args: { "aria-label": "Checkbox" },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Unchecked: Story = {};

export const Checked: Story = {
  args: { defaultChecked: true },
};

export const Disabled: Story = {
  args: { disabled: true },
};

export const DisabledChecked: Story = {
  args: { disabled: true, defaultChecked: true },
};
