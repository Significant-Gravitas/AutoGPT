import type { Meta, StoryObj } from "@storybook/react";
import { Status, StatusType } from "./Status";

const meta = {
  title: "AGPT UI/Status",
  component: Status,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    status: {
      control: "select",
      options: ["draft", "awaiting_review", "approved", "rejected"],
    },
  },
} satisfies Meta<typeof Status>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Draft: Story = {
  args: {
    status: "draft" as StatusType,
  },
};

export const AwaitingReview: Story = {
  args: {
    status: "awaiting_review" as StatusType,
  },
};

export const Approved: Story = {
  args: {
    status: "approved" as StatusType,
  },
};

export const Rejected: Story = {
  args: {
    status: "rejected" as StatusType,
  },
};

export const AllStatuses: Story = {
  args: {
    status: "draft" as StatusType,
  },
  render: () => (
    <div className="flex flex-col gap-4">
      <Status status="draft" />
      <Status status="awaiting_review" />
      <Status status="approved" />
      <Status status="rejected" />
    </div>
  ),
};
