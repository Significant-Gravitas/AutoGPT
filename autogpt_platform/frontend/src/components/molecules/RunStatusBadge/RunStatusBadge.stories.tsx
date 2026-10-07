import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { RunStatusBadge } from "./RunStatusBadge";

const ALL_STATUSES = [
  "COMPLETED",
  "FAILED",
  "RUNNING",
  "QUEUED",
  "REVIEW",
  "TERMINATED",
  "INCOMPLETE",
];

const meta = {
  title: "Molecules/RunStatusBadge",
  component: RunStatusBadge,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "A small Badge naming an agent run's status. Known statuses get a friendly label and a tone (success, error, warning, info); the lookup is case-insensitive and an unknown status falls back to its raw value in the neutral info tone.",
      },
    },
  },
  argTypes: {
    status: {
      control: "select",
      options: [...ALL_STATUSES, "completed", "SOMETHING_NEW"],
      description: "Run status as the backend reports it",
    },
  },
  args: { status: "COMPLETED" },
} satisfies Meta<typeof RunStatusBadge>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Completed: Story = {};

export const Failed: Story = { args: { status: "FAILED" } };

export const Running: Story = { args: { status: "RUNNING" } };

export const Queued: Story = { args: { status: "QUEUED" } };

export const WaitingForReview: Story = { args: { status: "REVIEW" } };

export const Stopped: Story = { args: { status: "TERMINATED" } };

export const Incomplete: Story = { args: { status: "INCOMPLETE" } };

export const LowercaseStatus: Story = {
  args: { status: "completed" },
  parameters: {
    docs: {
      description: {
        story: "The lookup is case-insensitive: `completed` reads Completed.",
      },
    },
  },
};

export const UnknownStatus: Story = {
  args: { status: "SOMETHING_NEW" },
  parameters: {
    docs: {
      description: {
        story:
          "A status the badge does not know shows its raw value in the neutral info tone.",
      },
    },
  },
};

export const AllStatuses: Story = {
  render: renderAllStatuses,
};

function renderAllStatuses() {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {ALL_STATUSES.map((status) => (
        <RunStatusBadge key={status} status={status} />
      ))}
    </div>
  );
}
