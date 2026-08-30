import type { Meta, StoryObj } from "@storybook/nextjs";
import { Orb } from "./Orb";

const meta: Meta<typeof Orb> = {
  title: "Atoms/Orb",
  component: Orb,
  parameters: { layout: "centered" },
};

export default meta;
type Story = StoryObj<typeof Orb>;

export const Default: Story = {
  args: { size: 20 },
};

export const AllVariants: Story = {
  render: () => (
    <div className="flex items-center gap-6 text-zinc-500">
      {(["S1", "S2", "S3", "S4", "S5"] as const).map((variant) => (
        <div key={variant} className="flex flex-col items-center gap-2">
          <Orb variant={variant} size={24} />
          <span className="text-xs text-zinc-400">{variant}</span>
        </div>
      ))}
    </div>
  ),
};

export const Tinted: Story = {
  render: () => (
    <div className="flex items-center gap-6">
      <Orb size={24} className="text-violet-500" />
      <Orb size={24} className="text-sky-500" />
      <Orb size={24} className="text-amber-500" />
    </div>
  ),
};
