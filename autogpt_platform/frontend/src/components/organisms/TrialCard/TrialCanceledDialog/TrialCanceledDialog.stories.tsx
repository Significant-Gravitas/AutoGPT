import type { Meta, StoryObj } from "@storybook/nextjs";
import { TrialCanceledDialog } from "./TrialCanceledDialog";

const DAY_MS = 24 * 60 * 60 * 1000;

const meta = {
  title: "Organisms/TrialCanceledDialog",
  component: TrialCanceledDialog,
  args: {
    trial: {
      active: true,
      status: "trialing",
      ends_at: new Date(Date.now() + 5 * DAY_MS),
      cancel_at_period_end: true,
      cancel_keeps_access: true,
      offer: {
        token: "a".repeat(64),
        version: "storybook-trial",
        duration_days: 7,
        tier: "PRO",
        billing_cycle: "monthly",
        unit_amount: 5000,
        currency: "usd",
        onboarding_credit_amount: 300,
      },
    },
    isOpen: true,
    isResuming: false,
    onResume: function onResume() {},
    onSubscribe: function onSubscribe() {},
    onClose: function onClose() {},
  },
} satisfies Meta<typeof TrialCanceledDialog>;

export default meta;
type Story = StoryObj<typeof meta>;

export const DaysLeft: Story = {};

export const UnderADay: Story = {
  args: {
    trial: {
      ...meta.args.trial,
      ends_at: new Date(Date.now() + 3 * 60 * 60 * 1000),
    },
  },
};

export const Resuming: Story = {
  args: { isResuming: true },
};
