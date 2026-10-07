import { Text } from "@/components/atoms/Text/Text";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { fn } from "storybook/test";
import { DateTimeInput } from "./DateTimeInput";

const meta = {
  title: "Atoms/DateTimeInput",
  component: DateTimeInput,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="w-80">
        <Story />
      </div>
    ),
  ],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          'Date and time picker: a trigger that opens a calendar plus a time field in a popover. `value` and `onChange` use a local `"YYYY-MM-DDTHH:MM"` string; the trigger shows the date and time in the browser locale. Supports a label with hint, error message, disabled and readonly states, and `default`/`small` sizes.',
      },
    },
  },
  argTypes: {
    value: {
      control: "text",
      description: 'Selected date and time as "YYYY-MM-DDTHH:MM"',
    },
    placeholder: {
      control: "text",
      description: 'Trigger text when empty. Defaults to "Pick date and time"',
    },
    label: { control: "text", description: "Label text" },
    hint: { control: "text", description: "Hint shown next to the label" },
    hideLabel: {
      control: "boolean",
      description: "Hide the visible label and use it as the aria-label",
    },
    error: { control: "text", description: "Error message under the input" },
    disabled: { control: "boolean", description: "Disable the trigger" },
    readonly: {
      control: "boolean",
      description: "Show the value but block opening the picker",
    },
    size: {
      control: "select",
      options: ["sm", "md", "lg"],
      description: "Trigger height",
    },
  },
  args: {
    id: "run-at",
    label: "Run at",
    onChange: fn(),
    size: "lg",
    disabled: false,
    readonly: false,
    hideLabel: false,
  },
} satisfies Meta<typeof DateTimeInput>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithValue: Story = {
  args: { value: "2026-01-15T10:30" },
};

export const WithHint: Story = {
  args: { value: "2026-01-15T10:30", hint: "Your local time" },
};

export const CustomPlaceholder: Story = {
  args: { placeholder: "Schedule the first run" },
};

export const HiddenLabel: Story = {
  args: { hideLabel: true, value: "2026-01-15T10:30" },
};

export const WithError: Story = {
  args: { error: "Pick a time in the future" },
};

export const Disabled: Story = {
  args: { disabled: true, value: "2026-01-15T10:30" },
};

export const Readonly: Story = {
  args: { readonly: true, value: "2026-01-15T10:30" },
};

export const Small: Story = {
  args: { size: "md", value: "2026-01-15T10:30" },
};

export const Interactive: Story = {
  render: renderInteractive,
};

export const AllSizes: Story = {
  render: renderAllSizes,
};

function InteractiveDateTimeInput() {
  const [value, setValue] = useState<string | undefined>("2026-01-15T10:30");

  return (
    <div className="flex flex-col gap-2">
      <DateTimeInput
        id="interactive-run-at"
        label="Run at"
        value={value}
        onChange={setValue}
      />
      <Text variant="small" tone="muted">
        Value: {value ?? "none"}
      </Text>
    </div>
  );
}

function renderInteractive() {
  return <InteractiveDateTimeInput />;
}

function renderAllSizes() {
  return (
    <div className="flex flex-col gap-4">
      <DateTimeInput
        id="size-default"
        label="Default"
        size="lg"
        value="2026-01-15T10:30"
      />
      <DateTimeInput
        id="size-small"
        label="Small"
        size="md"
        value="2026-01-15T10:30"
      />
    </div>
  );
}
