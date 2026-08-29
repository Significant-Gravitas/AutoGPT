import type { Meta, StoryObj } from "@storybook/nextjs";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { Table } from "./Table";

const meta = {
  title: "Molecules/Table",
  component: Table,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    allowAddRow: {
      control: "boolean",
      description: "Whether to show the Add row button",
    },
    allowDeleteRow: {
      control: "boolean",
      description: "Whether to show delete buttons for each row",
    },
    readOnly: {
      control: "boolean",
      description:
        "Whether the table is read-only (renders text instead of inputs)",
    },
    addRowLabel: {
      control: "text",
      description: "Label for the Add row button",
    },
  },
} satisfies Meta<typeof Table>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    columns: ["name", "email", "role"],
    allowAddRow: true,
    allowDeleteRow: true,
  },
};

export const WithDefaultValues: Story = {
  args: {
    columns: ["name", "email", "role"],
    defaultValues: [
      { name: "John Doe", email: "john@example.com", role: "Admin" },
      { name: "Jane Smith", email: "jane@example.com", role: "User" },
      { name: "Bob Wilson", email: "bob@example.com", role: "Editor" },
    ],
    allowAddRow: true,
    allowDeleteRow: true,
  },
};

export const ReadOnly: Story = {
  args: {
    columns: ["name", "email"],
    defaultValues: [
      { name: "John Doe", email: "john@example.com" },
      { name: "Jane Smith", email: "jane@example.com" },
    ],
    readOnly: true,
  },
};

export const NoAddOrDelete: Story = {
  args: {
    columns: ["name", "email"],
    defaultValues: [
      { name: "John Doe", email: "john@example.com" },
      { name: "Jane Smith", email: "jane@example.com" },
    ],
    allowAddRow: false,
    allowDeleteRow: false,
  },
};

export const SingleColumn: Story = {
  args: {
    columns: ["item"],
    allowAddRow: true,
    allowDeleteRow: true,
    addRowLabel: "Add item",
  },
};

export const CustomAddLabel: Story = {
  args: {
    columns: ["key", "value"],
    allowAddRow: true,
    allowDeleteRow: true,
    addRowLabel: "Add new entry",
  },
};

export const KeyValuePairs: Story = {
  args: {
    columns: ["key", "value"],
    defaultValues: [
      { key: "API_KEY", value: "sk-..." },
      { key: "DATABASE_URL", value: "postgres://..." },
    ],
    allowAddRow: true,
    allowDeleteRow: true,
    addRowLabel: "Add variable",
  },
};
