import type { Meta, StoryObj } from "@storybook/nextjs";
import { Badge } from "./Badge";

const meta: Meta<typeof Badge> = {
  title: "Atoms/Badge",
  tags: ["autodocs"],
  component: Badge,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Badge component for displaying status information with different variants for success, error, and info states.",
      },
    },
  },
  argTypes: {
    variant: {
      control: "select",
      options: ["success", "error", "info"],
      description: "Badge variant that determines color scheme",
    },
    children: {
      control: "text",
      description: "Badge content",
    },
    className: {
      control: "text",
      description: "Additional CSS classes",
    },
  },
  args: {
    variant: "success",
    children: "Success",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Success: Story = {
  args: {
    variant: "success",
    children: "Success",
  },
};

export const Error: Story = {
  args: {
    variant: "error",
    children: "Failed",
  },
};

export const Info: Story = {
  args: {
    variant: "info",
    children: "Stopped",
  },
};

export const AllVariants: Story = {
  render: renderAllVariants,
};

function renderAllVariants() {
  return (
    <div className="flex flex-wrap gap-4">
      <Badge variant="success">Success</Badge>
      <Badge variant="error">Failed</Badge>
      <Badge variant="info">Stopped</Badge>
      <Badge variant="info">Running</Badge>
      <Badge variant="success">Completed</Badge>
      <Badge variant="error">Error</Badge>
    </div>
  );
}
