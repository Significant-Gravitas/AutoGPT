import { Badge } from "@/components/atoms/Badge/Badge";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { FilterTable } from "./FilterTable";

const meta: Meta<typeof FilterTable> = {
  title: "Molecules/FilterTable",
  component: FilterTable,
  parameters: { layout: "padded" },
};

export default meta;
type Story = StoryObj<typeof FilterTable>;

const columns = [
  { key: "task", label: "Task name", width: "minmax(0,1.3fr)" },
  { key: "date", label: "Date", width: "minmax(0,0.6fr)" },
  { key: "status", label: "Status", width: "minmax(0,0.95fr)" },
  { key: "owner", label: "Owner", width: "minmax(0,0.9fr)" },
];

const filters = [
  { key: "todo", label: "To do", dot: "#f09a2f" },
  { key: "progress", label: "In Progress", dot: "#16a6c7" },
  { key: "done", label: "Completed", dot: "#25a878" },
];

function statusBadge(status: "todo" | "progress" | "done") {
  if (status === "done") {
    return (
      <Badge variant="success" size="small">
        Completed
      </Badge>
    );
  }
  if (status === "progress") {
    return (
      <Badge variant="info" size="small">
        In Progress
      </Badge>
    );
  }
  return (
    <Badge variant="warning" size="small">
      To do
    </Badge>
  );
}

const rows = [
  {
    id: "1",
    filterKey: "todo",
    cells: {
      task: <span className="truncate font-medium">Restock mango sorbet</span>,
      date: "Dec 03",
      status: statusBadge("todo"),
      owner: "Mango Moon Gelato",
    },
  },
  {
    id: "2",
    filterKey: "progress",
    cells: {
      task: <span className="truncate font-medium">Churn black sesame</span>,
      date: "Sep 22",
      status: statusBadge("progress"),
      owner: "Kumo Creamery",
    },
  },
  {
    id: "3",
    filterKey: "done",
    cells: {
      task: <span className="truncate font-medium">Order waffle cones</span>,
      date: "Apr 14",
      status: statusBadge("done"),
      owner: "Aurora Scoops",
    },
  },
];

export const Default: Story = {
  args: { columns, filters, rows, ariaLabel: "Tasks" },
};

export const WithoutFilters: Story = {
  args: { columns, rows, ariaLabel: "Tasks" },
};

export const Empty: Story = {
  args: {
    columns,
    filters,
    rows: [],
    emptyText: "Nothing delegated yet.",
    ariaLabel: "Tasks",
  },
};
