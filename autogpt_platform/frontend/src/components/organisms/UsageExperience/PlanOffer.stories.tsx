import type { Meta, StoryObj } from "@storybook/nextjs";
import { fn } from "storybook/test";
import { PlanOffer } from "./PlanOffer";
const meta = {
  title: "Organisms/UsageExperience/PlanOffer",
  component: PlanOffer,
  parameters: { layout: "centered" },
  args: { onUpgrade: fn() },
  decorators: [
    (Story) => (
      <div className="w-full max-w-md">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof PlanOffer>;
export default meta;
type Story = StoryObj<typeof meta>;
export const TrialToPro: Story = {
  args: { tier: "PRO", price: 5000, freshUsage: true },
};
export const ProToMax: Story = {
  args: { tier: "MAX", price: 32000, usageMultiplier: 8.5 },
};
export const AcceptedYearlyOffer: Story = {
  args: { tier: "PRO", price: 51000, cycle: "yearly", freshUsage: true },
};
