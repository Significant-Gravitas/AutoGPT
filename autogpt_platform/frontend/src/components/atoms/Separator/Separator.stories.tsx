import type { Meta, StoryObj } from "@storybook/nextjs";
import { Text } from "../Text/Text";
import { Separator } from "./Separator";

const meta: Meta<typeof Separator> = {
  title: "Atoms/Separator",
  component: Separator,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "A 1px rule between groups of content, horizontal or vertical. Decorative by default (hidden from assistive technology).",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Horizontal: Story = {
  render: function HorizontalStory() {
    return (
      <div className="w-64">
        <Text variant="body" className="py-2">
          Above
        </Text>
        <Separator />
        <Text variant="body" className="py-2">
          Below
        </Text>
      </div>
    );
  },
};

export const Vertical: Story = {
  render: function VerticalStory() {
    return (
      <div className="flex h-8 items-center gap-3">
        <Text variant="body" as="span">
          Left
        </Text>
        <Separator orientation="vertical" />
        <Text variant="body" as="span">
          Right
        </Text>
      </div>
    );
  },
};
