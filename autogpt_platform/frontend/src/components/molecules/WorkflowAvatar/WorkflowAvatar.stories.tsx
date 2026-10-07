import type { Meta, StoryObj } from "@storybook/nextjs";
import { Text } from "@/components/atoms/Text/Text";
import { WorkflowAvatar } from "./WorkflowAvatar";

const WORKFLOW_IMAGE = "/images/team-card-banner.jpg";

const meta = {
  title: "Molecules/WorkflowAvatar",
  component: WorkflowAvatar,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "A workflow's own picture in a round Avatar, falling back to the generated marble seeded by the workflow name. Two sizes: 18px for a row bullet and 36px for a task marker.",
      },
    },
  },
  argTypes: {
    size: {
      control: "inline-radio",
      options: [18, 36],
      description: "Rendered size in pixels",
    },
    imageUrl: {
      control: "text",
      description: "The workflow's image; empty falls back to the marble",
    },
  },
  args: { name: "Morning digest", imageUrl: WORKFLOW_IMAGE, size: 18 },
} satisfies Meta<typeof WorkflowAvatar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Large: Story = { args: { size: 36 } };

export const Fallback: Story = {
  args: { imageUrl: null, size: 36 },
  parameters: {
    docs: {
      description: {
        story:
          "No image: the marble generated from the workflow name. Different names give different marbles.",
      },
    },
  },
};

export const BrokenImage: Story = {
  args: { imageUrl: "/images/does-not-exist.png", size: 36 },
  parameters: {
    docs: {
      description: {
        story: "An image that fails to load falls back to the marble.",
      },
    },
  },
};

export const AllSizes: Story = {
  render: renderAllSizes,
};

const NAMES = ["Morning digest", "Lead finder", "Invoice chaser"];

function renderAllSizes() {
  return (
    <div className="flex flex-col gap-4">
      {([18, 36] as const).map((size) => (
        <div key={size} className="flex items-center gap-3">
          <Text variant="small" className="w-10 text-zinc-500">
            {size}px
          </Text>
          <WorkflowAvatar
            name="Morning digest"
            imageUrl={WORKFLOW_IMAGE}
            size={size}
          />
          {NAMES.map((name) => (
            <WorkflowAvatar
              key={name}
              name={name}
              imageUrl={null}
              size={size}
            />
          ))}
        </div>
      ))}
    </div>
  );
}
