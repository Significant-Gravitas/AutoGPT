import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { Checkbox } from "./Checkbox";

const meta: Meta<typeof Checkbox> = {
  title: "Atoms/Checkbox",
  component: Checkbox,
  tags: ["autodocs"],
  parameters: {
    a11y: { test: "error" },
    layout: "centered",
    docs: {
      description: {
        component:
          'Radix checkbox with an optional label, description and error. Pass `checked="indeterminate"` for a partial selection. Without a label, give it an `aria-label`.',
      },
    },
  },
  argTypes: {
    size: { control: "inline-radio", options: ["sm", "md"] },
    label: { control: "text" },
    description: { control: "text" },
    error: { control: "text" },
    disabled: { control: "boolean" },
    onCheckedChange: { action: "change" },
  },
  args: {
    label: "Email me about new runs",
    size: "sm",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Checked: Story = {
  args: { defaultChecked: true },
};

export const Indeterminate: Story = {
  args: { checked: "indeterminate", label: "Select all" },
};

export const WithDescription: Story = {
  args: {
    description: "We send at most one email a day.",
  },
};

export const WithError: Story = {
  args: {
    label: "I accept the terms",
    error: "You must accept the terms to continue.",
  },
};

export const Disabled: Story = {
  args: { disabled: true, defaultChecked: true },
};

export const Medium: Story = {
  args: {
    size: "md",
    defaultChecked: true,
    description: "The larger box for settings lists.",
  },
};

export const WithoutLabel: Story = {
  args: { label: undefined, "aria-label": "Select row" },
};

export const SelectAll: Story = {
  render: function SelectAllStory() {
    const options = ["Executions", "Schedules", "Webhooks"];
    const [selected, setSelected] = useState<string[]>(["Executions"]);
    const allState =
      selected.length === options.length
        ? true
        : selected.length > 0
          ? "indeterminate"
          : false;

    function handleToggleAll() {
      setSelected(selected.length === options.length ? [] : options);
    }

    function handleToggle(option: string) {
      setSelected((current) =>
        current.includes(option)
          ? current.filter((value) => value !== option)
          : [...current, option],
      );
    }

    return (
      <div className="flex flex-col gap-3">
        <Checkbox
          label="Select all"
          checked={allState}
          onCheckedChange={handleToggleAll}
        />
        <div className="flex flex-col gap-2 pl-6">
          {options.map((option) => (
            <Checkbox
              key={option}
              label={option}
              checked={selected.includes(option)}
              onCheckedChange={() => handleToggle(option)}
            />
          ))}
        </div>
      </div>
    );
  },
};
