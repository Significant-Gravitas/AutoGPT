import { Badge } from "@/components/atoms/Badge/Badge";
import {
  CheckmarkCircle02Icon,
  Clock01Icon,
  DashboardSquare01Icon,
  Progress02Icon,
  Task01Icon,
  UserIcon,
} from "@hugeicons/core-free-icons";
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
  {
    key: "task",
    label: "Task name",
    icon: Task01Icon,
    width: "minmax(0,1.3fr)",
  },
  { key: "date", label: "Date", icon: Clock01Icon, width: "minmax(0,0.6fr)" },
  {
    key: "status",
    label: "Status",
    icon: Progress02Icon,
    width: "minmax(0,0.95fr)",
  },
  { key: "owner", label: "Owner", icon: UserIcon, width: "minmax(0,0.9fr)" },
];

const filters = [
  { key: "todo", label: "To do", icon: Clock01Icon, dot: "#f09a2f" },
  { key: "progress", label: "In Progress", icon: Loading03Icon, dot: "#16a6c7" },
  {
    key: "done",
    label: "Completed",
    icon: CheckmarkCircle02Icon,
    dot: "#25a878",
  },
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
  args: {
    columns,
    filters,
    rows,
    allIcon: DashboardSquare01Icon,
    ariaLabel: "Tasks",
  },
};

export const WithFilterOverflow: Story = {
  args: {
    columns,
    filters,
    rows,
    allIcon: DashboardSquare01Icon,
    maxVisibleFilters: 2,
    ariaLabel: "Tasks",
  },
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
