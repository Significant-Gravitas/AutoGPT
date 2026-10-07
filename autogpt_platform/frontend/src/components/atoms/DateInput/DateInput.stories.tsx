import { Text } from "@/components/atoms/Text/Text";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { fn, userEvent, within } from "storybook/test";
import { DateInput } from "./DateInput";

const meta = {
  title: "Atoms/DateInput",
  component: DateInput,
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
          'Date picker that opens a calendar in a popover. `value` and `onChange` use a local `"YYYY-MM-DD"` string; the trigger shows the date in the browser locale. Supports a label (visible or hidden), error message, disabled and readonly states, and `default`/`small` sizes.',
      },
    },
  },
  argTypes: {
    value: {
      control: "text",
      description: 'Selected date as "YYYY-MM-DD"',
    },
    placeholder: {
      control: "text",
      description: 'Trigger text when empty. Defaults to "Pick a date"',
    },
    label: { control: "text", description: "Label text" },
    hideLabel: {
      control: "boolean",
      description: "Hide the visible label and use it as the aria-label",
    },
    error: { control: "text", description: "Error message under the input" },
    disabled: { control: "boolean", description: "Disable the trigger" },
    readonly: {
      control: "boolean",
      description: "Show the value but block opening the calendar",
    },
    size: {
      control: "select",
      options: ["sm", "md", "lg"],
      description: "Trigger height",
    },
  },
  args: {
    id: "start-date",
    label: "Start date",
    onChange: fn(),
    size: "lg",
    disabled: false,
    readonly: false,
    hideLabel: false,
  },
} satisfies Meta<typeof DateInput>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithValue: Story = {
  args: { value: "2026-01-15" },
};

export const CustomPlaceholder: Story = {
  args: { placeholder: "Select a start date" },
};

export const HiddenLabel: Story = {
  args: { hideLabel: true, value: "2026-01-15" },
};

export const WithError: Story = {
  args: { error: "Start date is required" },
};

export const Disabled: Story = {
  args: { disabled: true, value: "2026-01-15" },
};

export const Readonly: Story = {
  args: { readonly: true, value: "2026-01-15" },
};

export const Small: Story = {
  args: { size: "md", value: "2026-01-15" },
};

export const Open: Story = {
  args: { value: "2026-01-15" },
  play: async ({ canvasElement }) => {
    await userEvent.click(within(canvasElement).getByRole("button"));
  },
};

export const Interactive: Story = {
  render: renderInteractive,
};

export const AllSizes: Story = {
  render: renderAllSizes,
};

function InteractiveDateInput() {
  const [value, setValue] = useState<string | undefined>("2026-01-15");

  return (
    <div className="flex flex-col gap-2">
      <DateInput
        id="interactive-date"
        label="Start date"
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
  return <InteractiveDateInput />;
}

function renderAllSizes() {
  return (
    <div className="flex flex-col gap-4">
      <DateInput
        id="size-default"
        label="Default"
        size="lg"
        value="2026-01-15"
      />
      <DateInput id="size-small" label="Small" size="md" value="2026-01-15" />
    </div>
  );
}
