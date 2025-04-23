import type { Meta, StoryObj } from "@storybook/react";

import {
  Sheet,
  SheetTrigger,
  SheetContent,
  SheetHeader,
  SheetFooter,
  SheetTitle,
  SheetDescription,
} from "./sheet";
import { Button } from "./button";

const meta = {
  title: "UI/Sheet",
  component: Sheet,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof Sheet>;

export default meta;
type Story = StoryObj<typeof meta>;

const SheetDemo = ({ side }: { side: "top" | "right" | "bottom" | "left" }) => (
  <Sheet>
    <SheetTrigger asChild>
      <Button variant="outline">Open Sheet</Button>
    </SheetTrigger>
    <SheetContent side={side}>
      <SheetHeader>
        <SheetTitle>Sheet Title</SheetTitle>
        <SheetDescription>
          This is a description of the sheet content.
        </SheetDescription>
      </SheetHeader>
      <div className="py-4">Sheet content goes here.</div>
      <SheetFooter>
        <Button>Save changes</Button>
      </SheetFooter>
    </SheetContent>
  </Sheet>
);

export const Default: Story = {
  render: () => <SheetDemo side="right" />,
};

export const Left: Story = {
  render: () => <SheetDemo side="left" />,
};

export const Top: Story = {
  render: () => <SheetDemo side="top" />,
};

export const Bottom: Story = {
  render: () => <SheetDemo side="bottom" />,
};

export const CustomContent: Story = {
  render: () => (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="outline">Open Custom Sheet</Button>
      </SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>Custom Sheet</SheetTitle>
        </SheetHeader>
        <div className="py-4">
          <p>This sheet has custom content.</p>
          <ul className="list-disc pl-4 pt-2">
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
          </ul>
        </div>
      </SheetContent>
    </Sheet>
  ),
};
