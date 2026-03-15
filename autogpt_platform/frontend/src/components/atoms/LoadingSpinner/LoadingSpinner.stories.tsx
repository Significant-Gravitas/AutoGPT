import type { Meta, StoryObj } from "@storybook/nextjs";
import { LoadingSpinner } from "./LoadingSpinner";

const meta: Meta<typeof LoadingSpinner> = {
  title: "Atoms/LoadingSpinner",
  component: LoadingSpinner,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Animated loading indicator using the Phosphor CircleNotch icon. Provide a `size` prop or custom classes to fit different contexts.",
      },
    },
  },
  argTypes: {
    size: {
      control: "select",
      options: ["small", "medium", "large"],
      description: "Spinner size preset",
    },
    className: {
      control: "text",
      description: "Additional CSS classes to customize color or layout",
    },
  },
  args: {
    size: "medium",
    className: "text-indigo-500",
    role: "status",
    "aria-label": "loading",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Small: Story = {
  args: {
    size: "small",
  },
};

export const Large: Story = {
  args: {
    size: "large",
  },
};

export const CustomColor: Story = {
  args: {
    className: "text-emerald-500",
  },
};

export const Cover: Story = {
  args: {
    cover: true,
  },
};

export const AllSizes: Story = {
  render: renderAllSizes,
};

function renderAllSizes() {
  return (
    <div className="flex items-center gap-8 text-indigo-500">
      <div className="flex flex-col items-center gap-2">
        <LoadingSpinner size="small" aria-label="loading-small" />
        <span className="text-xs capitalize text-zinc-500">Small</span>
      </div>
      <div className="flex flex-col items-center gap-2">
        <LoadingSpinner size="medium" aria-label="loading-medium" />
        <span className="text-xs capitalize text-zinc-500">Medium</span>
      </div>
      <div className="flex flex-col items-center gap-2">
        <LoadingSpinner size="large" aria-label="loading-large" />
        <span className="text-xs capitalize text-zinc-500">Large</span>
      </div>
    </div>
  );
}
