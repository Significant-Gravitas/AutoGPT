import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { NumberTicker } from "./NumberTicker";

const meta: Meta<typeof NumberTicker> = {
  title: "Atoms/NumberTicker",
  tags: ["autodocs"],
  component: NumberTicker,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "A count whose changed digits roll in (450ms, out-expo). Digits are equal width and hidden from screen readers, which read the plain value. Stops animating under reduced motion.",
      },
    },
  },
  args: { value: 42 },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Large: Story = {
  args: { value: 48240 },
};

export const Interactive: Story = {
  render: function InteractiveStory() {
    const [value, setValue] = useState(7);
    return (
      <div className="flex flex-col items-center gap-4">
        <Text variant="h3" tone="ink">
          <NumberTicker value={value} />
        </Text>
        <div className="flex gap-2">
          <Button
            variant="secondary"
            size="small"
            onClick={() => setValue((v) => Math.max(0, v - 1))}
          >
            Minus one
          </Button>
          <Button
            variant="secondary"
            size="small"
            onClick={() => setValue((v) => v + 1)}
          >
            Plus one
          </Button>
          <Button
            variant="secondary"
            size="small"
            onClick={() => setValue((v) => v + 137)}
          >
            Plus 137
          </Button>
        </div>
      </div>
    );
  },
};
