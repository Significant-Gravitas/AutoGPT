import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { MultiSelect, type MultiSelectOption } from "./MultiSelect";

const OPTIONS: MultiSelectOption[] = [
  { value: "executions", label: "Executions" },
  { value: "schedules", label: "Schedules" },
  { value: "webhooks", label: "Webhooks" },
  { value: "credits", label: "Credits", description: "Balance changes" },
  { value: "archived", label: "Archived", disabled: true },
];

const meta: Meta<typeof MultiSelect> = {
  title: "Molecules/MultiSelect",
  component: MultiSelect,
  tags: ["autodocs"],
  parameters: {
    a11y: { test: "error" },
    layout: "centered",
    docs: {
      description: {
        component:
          "Pick several options from a list. Selected options show as chips in the field; typing filters the list by label. Controlled through `value` and `onValueChange`.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="w-96">
        <Story />
      </div>
    ),
  ],
  args: {
    options: OPTIONS,
    placeholder: "Select notifications...",
    "aria-label": "Notifications",
  },
  argTypes: {
    disabled: { control: "boolean" },
    placeholder: { control: "text" },
    emptyMessage: { control: "text" },
    onValueChange: { action: "change" },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

function ControlledExample(args: React.ComponentProps<typeof MultiSelect>) {
  const [value, setValue] = useState(args.value ?? []);
  return (
    <MultiSelect
      {...args}
      value={value}
      onValueChange={(next) => {
        setValue(next);
        args.onValueChange(next);
      }}
    />
  );
}

export const Default: Story = {
  args: { value: [] },
  render: (args) => <ControlledExample {...args} />,
};

export const WithSelection: Story = {
  args: { value: ["executions", "webhooks"] },
  render: (args) => <ControlledExample {...args} />,
};

export const Disabled: Story = {
  args: { value: ["schedules"], disabled: true },
  render: (args) => <ControlledExample {...args} />,
};
