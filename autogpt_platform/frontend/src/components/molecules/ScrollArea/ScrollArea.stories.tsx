import type { Meta, StoryObj } from "@storybook/nextjs";
import { ScrollArea } from "./ScrollArea";

const meta: Meta<typeof ScrollArea> = {
  title: "Molecules/ScrollArea",
  component: ScrollArea,
  parameters: { layout: "centered" },
};

export default meta;
type Story = StoryObj<typeof ScrollArea>;

const rows = Array.from({ length: 30 }, (_, i) => `Row ${i + 1}`);

export const Vertical: Story = {
  render: () => (
    <ScrollArea className="h-48 w-56 rounded-medium border border-border-base">
      <div className="p-3">
        {rows.map((row) => (
          <div key={row} className="py-1">
            {row}
          </div>
        ))}
      </div>
    </ScrollArea>
  ),
};

export const Horizontal: Story = {
  render: () => (
    <ScrollArea
      orientation="horizontal"
      className="w-72 rounded-medium border border-border-base"
    >
      <div className="flex w-max gap-3 p-3">
        {rows.slice(0, 12).map((row) => (
          <div
            key={row}
            className="h-16 w-24 shrink-0 rounded-medium bg-surface-muted"
          />
        ))}
      </div>
    </ScrollArea>
  ),
};
