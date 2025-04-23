import type { Meta, StoryObj } from "@storybook/react";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "./collapsible";
import { Button } from "./button";

const meta = {
  title: "UI/Collapsible",
  component: Collapsible,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof Collapsible>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => (
    <Collapsible>
      <CollapsibleTrigger asChild>
        <Button variant="outline">Toggle</Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 rounded bg-gray-100 p-4">
        <p>This is the collapsible content.</p>
      </CollapsibleContent>
    </Collapsible>
  ),
};

export const OpenByDefault: Story = {
  render: () => (
    <Collapsible defaultOpen>
      <CollapsibleTrigger asChild>
        <Button variant="outline">Toggle</Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 rounded bg-gray-100 p-4">
        <p>This collapsible is open by default.</p>
      </CollapsibleContent>
    </Collapsible>
  ),
};

export const CustomTrigger: Story = {
  render: () => (
    <Collapsible>
      <CollapsibleTrigger asChild>
        <Button variant="ghost">
          <span>🔽</span> Click to expand
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 rounded bg-gray-100 p-4">
        <p>Custom trigger example.</p>
      </CollapsibleContent>
    </Collapsible>
  ),
};

export const NestedContent: Story = {
  render: () => (
    <Collapsible>
      <CollapsibleTrigger asChild>
        <Button variant="outline">Toggle Nested Content</Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 rounded bg-gray-100 p-4">
        <h3 className="mb-2 font-bold">Main Content</h3>
        <p className="mb-2">This is the main collapsible content.</p>
        <Collapsible>
          <CollapsibleTrigger asChild>
            <Button variant="secondary" size="sm">
              Toggle Nested
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-2 rounded bg-gray-200 p-2">
            <p>This is nested collapsible content.</p>
          </CollapsibleContent>
        </Collapsible>
      </CollapsibleContent>
    </Collapsible>
  ),
};
