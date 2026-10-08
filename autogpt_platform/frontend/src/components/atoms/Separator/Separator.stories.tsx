import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Text } from "../Text/Text";
import { Separator } from "./Separator";

const meta: Meta<typeof Separator> = {
  title: "Atoms/Separator",
  component: Separator,
  tags: ["autodocs"],
  parameters: {
    a11y: { test: "error" },
    layout: "centered",
    docs: {
      description: {
        component:
          "A one-pixel rule on Kobra's Separator (`bg-foreground/15`). Decorative by default, so screen readers skip it; pass `decorative={false}` when it separates content semantically. Vertical separators stretch to their flex row.",
      },
    },
  },
  argTypes: {
    orientation: {
      control: "inline-radio",
      options: ["horizontal", "vertical"],
    },
    decorative: { control: "boolean" },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Horizontal: Story = {
  render: function HorizontalStory(args) {
    return (
      <div className="w-72">
        <Text variant="body-medium">Account</Text>
        <Text variant="small" tone="secondary">
          Name, email and password.
        </Text>
        <Separator {...args} className="my-4" />
        <Text variant="body-medium">Billing</Text>
        <Text variant="small" tone="secondary">
          Plan, invoices and payment method.
        </Text>
      </div>
    );
  },
};

export const Vertical: Story = {
  render: function VerticalStory() {
    return (
      <div className="flex h-5 items-center gap-3">
        <Text variant="body">Docs</Text>
        <Separator orientation="vertical" />
        <Text variant="body">Source</Text>
        <Separator orientation="vertical" />
        <Text variant="body">Changelog</Text>
      </div>
    );
  },
};

export const Semantic: Story = {
  render: function SemanticStory() {
    return (
      <div className="w-72">
        <Text variant="body">Above</Text>
        <Separator decorative={false} className="my-3" />
        <Text variant="body">Below</Text>
      </div>
    );
  },
};
