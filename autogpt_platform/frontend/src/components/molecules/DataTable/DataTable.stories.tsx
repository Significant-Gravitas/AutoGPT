import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import { DataTable } from "./DataTable";
import type { DataTableColumn, DataTableSort } from "./helpers";

interface Execution {
  id: string;
  agent: string;
  user: string;
  status: "completed" | "failed" | "running";
  credits: number;
  startedAt: Date;
}

const EXECUTIONS: Execution[] = [
  {
    id: "exec-1",
    agent: "Weekly digest",
    user: "ana@example.com",
    status: "completed",
    credits: 42,
    startedAt: new Date("2026-10-01T09:00:00Z"),
  },
  {
    id: "exec-2",
    agent: "Lead enrichment",
    user: "ben@example.com",
    status: "failed",
    credits: 7,
    startedAt: new Date("2026-10-02T14:30:00Z"),
  },
  {
    id: "exec-3",
    agent: "Support triage",
    user: "chen@example.com",
    status: "running",
    credits: 120,
    startedAt: new Date("2026-10-03T08:15:00Z"),
  },
  {
    id: "exec-4",
    agent: "Blog drafter",
    user: "dee@example.com",
    status: "completed",
    credits: 18,
    startedAt: new Date("2026-09-28T17:45:00Z"),
  },
];

const STATUS_VARIANT = {
  completed: "success",
  failed: "error",
  running: "info",
} as const;

const COLUMNS: DataTableColumn<Execution>[] = [
  {
    key: "agent",
    header: "Agent",
    cell: (row) => row.agent,
    sortValue: (row) => row.agent,
  },
  {
    key: "user",
    header: "User",
    cell: (row) => (
      <Text variant="body" as="span" tone="secondary" unmask={false}>
        {row.user}
      </Text>
    ),
  },
  {
    key: "status",
    header: "Status",
    cell: (row) => (
      <Badge variant={STATUS_VARIANT[row.status]}>{row.status}</Badge>
    ),
    sortValue: (row) => row.status,
  },
  {
    key: "credits",
    header: "Credits",
    align: "right",
    cell: (row) => row.credits.toLocaleString(),
    sortValue: (row) => row.credits,
  },
  {
    key: "startedAt",
    header: "Started",
    cell: (row) => row.startedAt.toLocaleString("en-US"),
    sortValue: (row) => row.startedAt,
  },
];

const meta: Meta<typeof DataTable<Execution>> = {
  title: "Molecules/DataTable",
  component: DataTable,
  tags: ["autodocs"],
  parameters: {
    // Known axe findings, mostly colour contrast (DESIGN.md, "Story tests").
    // Back to "error" once they are fixed.
    a11y: { test: "todo" },
    layout: "padded",
    docs: {
      description: {
        component:
          "A read-only table typed by a `columns` array. A column with `sortValue` gets a sort button and `aria-sort`; sorting is local unless `manualSorting` is set. Supports loading skeleton rows, an empty state and an optional row click (keyboard: Enter or Space).",
      },
    },
  },
  args: {
    columns: COLUMNS,
    rows: EXECUTIONS,
    getRowKey: (row: Execution) => row.id,
    caption: "Recent executions",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const DefaultSorted: Story = {
  args: { defaultSort: { key: "credits", direction: "desc" } },
};

export const Loading: Story = {
  args: { isLoading: true, loadingRowCount: 4 },
};

export const Empty: Story = {
  args: { rows: [], emptyState: "No executions in this period" },
};

export const ClickableRows: Story = {
  render: function ClickableRowsStory(args) {
    const [selected, setSelected] = useState<string | null>(null);
    return (
      <div className="flex flex-col gap-3">
        <DataTable {...args} onRowClick={(row) => setSelected(row.agent)} />
        <Text variant="small" tone="secondary">
          {selected ? `Opened ${selected}` : "Click or press Enter on a row"}
        </Text>
      </div>
    );
  },
};

export const ServerSorted: Story = {
  render: function ServerSortedStory(args) {
    const [sort, setSort] = useState<DataTableSort | null>(null);
    return (
      <div className="flex flex-col gap-3">
        <DataTable {...args} sort={sort} onSortChange={setSort} manualSorting />
        <Text variant="small" tone="secondary">
          {sort
            ? `Would fetch sorted by ${sort.key} ${sort.direction}`
            : "Unsorted"}
        </Text>
      </div>
    );
  },
};
