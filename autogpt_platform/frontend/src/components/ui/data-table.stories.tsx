import type { Meta, StoryObj } from "@storybook/react";
import { DataTable } from "./data-table";
import { Button } from "./button";

const meta = {
  title: "UI/DataTable",
  component: DataTable,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof DataTable>;

export default meta;
type Story = StoryObj<typeof meta>;

const sampleData = [
  { id: 1, name: "John Doe", age: 30, city: "New York" },
  { id: 2, name: "Jane Smith", age: 25, city: "Los Angeles" },
  { id: 3, name: "Bob Johnson", age: 35, city: "Chicago" },
];

const sampleColumns = [
  { accessorKey: "name", header: "Name" },
  { accessorKey: "age", header: "Age" },
  { accessorKey: "city", header: "City" },
];

export const Default: Story = {
  args: {
    columns: sampleColumns,
    data: sampleData,
    filterPlaceholder: "Filter by name...",
    filterColumn: "name",
  },
};

export const WithGlobalActions: Story = {
  args: {
    ...Default.args,
    globalActions: [
      {
        component: <Button>Delete Selected</Button>,
        action: async (rows) => {
          console.log("Deleting:", rows);
        },
      },
    ],
  },
};

export const NoResults: Story = {
  args: {
    ...Default.args,
    data: [],
  },
};

export const CustomFilterPlaceholder: Story = {
  args: {
    ...Default.args,
    filterPlaceholder: "Search for a user...",
  },
};

export const WithoutFilter: Story = {
  args: {
    columns: sampleColumns,
    data: sampleData,
    filterPlaceholder: "",
  },
};
