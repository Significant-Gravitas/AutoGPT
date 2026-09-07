import type { Meta, StoryObj } from "@storybook/nextjs";
import { TrialStatus } from "./TrialStatus";

const meta = {
  title: "Organisms/TrialStatus",
  component: TrialStatus,
  args: {
    trial: {
      active: true,
      ends_at: new Date("2030-09-17T15:00:00Z"),
      allowance_used_percent: 42.4,
      cancel_at_period_end: false,
      offer: {
        token: "a".repeat(64),
        version: "storybook-trial",
        duration_days: 7,
        tier: "PRO",
        billing_cycle: "monthly",
        unit_amount: 2000,
        currency: "usd",
        onboarding_credit_amount: 300,
      },
    },
    isCanceling: false,
    onCancel: function onCancel() {},
  },
} satisfies Meta<typeof TrialStatus>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Active: Story = {};

export const CancellationScheduled: Story = {
  args: {
    trial: { ...meta.args.trial, cancel_at_period_end: true },
  },
};
