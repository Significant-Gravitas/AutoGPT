import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Card } from "./Card";

const meta = {
  title: "Atoms/Card",
  component: Card,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "Kobra's Card surface (bg-card, rounded-xl, a hairline ring) with 1.5rem padding and children flowing normally. It has no variants: pass `className` to change width, padding or layout.",
      },
    },
  },
  argTypes: {
    className: {
      control: "text",
      description: "Extra classes merged over the base surface styles",
    },
  },
  args: {
    className: "w-80",
    children: (
      <Text variant="body">
        A card groups related content on a single surface.
      </Text>
    ),
  },
} satisfies Meta<typeof Card>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithHeaderAndActions: Story = {
  args: {
    children: (
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <Text variant="h5" as="h2">
            Weekly report agent
          </Text>
          <Text variant="body" tone="secondary">
            Summarises your team&apos;s pull requests every Monday at 09:00.
          </Text>
        </div>
        <div className="flex gap-2">
          <Button variant="primary" size="md">
            Run agent
          </Button>
          <Button variant="secondary" size="md">
            Edit
          </Button>
        </div>
      </div>
    ),
  },
};

export const Compact: Story = {
  args: {
    className: "w-80 p-4",
    children: (
      <Text variant="body" tone="secondary">
        Compact card with 1rem padding.
      </Text>
    ),
  },
};

export const Empty: Story = {
  args: {
    className: "h-32 w-80",
    children: null,
  },
};

export const Grid: Story = {
  render: renderGrid,
};

function renderGrid() {
  const stats = [
    { label: "Runs this week", value: "128" },
    { label: "Success rate", value: "97%" },
    { label: "Credits used", value: "2,450" },
  ];

  return (
    <div className="grid grid-cols-3 gap-4">
      {stats.map((stat) => (
        <Card key={stat.label} className="w-48">
          <Text variant="small" tone="muted">
            {stat.label}
          </Text>
          <Text variant="h4" as="p">
            {stat.value}
          </Text>
        </Card>
      ))}
    </div>
  );
}
