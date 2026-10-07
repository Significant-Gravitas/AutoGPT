import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Sheet } from "./Sheet";

const meta: Meta<typeof Sheet> = {
  title: "Molecules/Sheet",
  component: Sheet,
  tags: ["autodocs"],
  parameters: {
    a11y: { test: "error" },
    layout: "centered",
    docs: {
      description: {
        component:
          "A panel that slides in from an edge of the screen, on Radix Dialog. `title` is required as the accessible name; hide it visually with `hideTitle`. Control it with `open`/`onOpenChange` or pass a `trigger`.",
      },
    },
  },
  argTypes: {
    side: {
      control: "inline-radio",
      options: ["right", "left", "top", "bottom"],
    },
    hideTitle: { control: "boolean" },
    hideDescription: { control: "boolean" },
  },
  args: {
    title: "Run output",
    description: "The latest output of this run.",
    side: "right",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

function Body() {
  return (
    <>
      <Text variant="body">
        Weekly signups rose 12% on the previous week, led by organic traffic.
      </Text>
      <Text variant="body" tone="secondary">
        Activation held at 68%. Referral signups doubled after the new invite
        flow shipped on Tuesday.
      </Text>
    </>
  );
}

export const Right: Story = {
  args: {
    trigger: <Button size="small">Open sheet</Button>,
    children: <Body />,
  },
};

export const Left: Story = {
  args: {
    side: "left",
    trigger: <Button size="small">Open from the left</Button>,
    children: <Body />,
  },
};

export const Top: Story = {
  args: {
    side: "top",
    trigger: <Button size="small">Open from the top</Button>,
    children: <Body />,
  },
};

export const Bottom: Story = {
  args: {
    side: "bottom",
    trigger: <Button size="small">Open from the bottom</Button>,
    children: <Body />,
  },
};

export const OpenWithFooter: Story = {
  args: {
    defaultOpen: true,
    trigger: <Button size="small">Open sheet</Button>,
    children: <Body />,
    footer: (
      <>
        <Button variant="secondary" size="small">
          Cancel
        </Button>
        <Button size="small">Save</Button>
      </>
    ),
  },
};

export const WithHeaderActions: Story = {
  args: {
    defaultOpen: true,
    trigger: <Button size="small">Open sheet</Button>,
    actions: (
      <Button variant="secondary" size="small">
        Export CSV
      </Button>
    ),
    children: <Body />,
  },
};

export const HiddenTitle: Story = {
  args: {
    defaultOpen: true,
    hideTitle: true,
    description: undefined,
    trigger: <Button size="small">Open sheet</Button>,
    children: <Body />,
  },
};

export const Controlled: Story = {
  render: function ControlledStory(args) {
    const [open, setOpen] = useState(false);
    return (
      <>
        <Button size="small" onClick={() => setOpen(true)}>
          Open controlled sheet
        </Button>
        <Sheet {...args} open={open} onOpenChange={setOpen}>
          <Body />
        </Sheet>
      </>
    );
  },
};
