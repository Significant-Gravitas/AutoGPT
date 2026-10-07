import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { fn } from "storybook/test";
import { TrialOffer } from "./TrialOffer";

const OFFER = {
  token: "a".repeat(64),
  version: "storybook-trial",
  duration_days: 7,
  tier: "PRO",
  billing_cycle: "monthly",
  unit_amount: 2000,
  currency: "usd",
  onboarding_credit_amount: 300,
} as const;

const meta = {
  title: "Organisms/TrialOffer",
  component: TrialOffer,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="w-full max-w-3xl">
        <Story />
      </div>
    ),
  ],
  parameters: {
    layout: "padded",
    // Known axe findings, mostly colour contrast (DESIGN.md, "Story tests").
    // Back to "error" once they are fixed.
    a11y: { test: "todo" },
    docs: {
      description: {
        component:
          "The trial offer body inside TrialCard: plan and duration, the price it converts to, and the Start trial button. Renders nothing when the trial has no offer.",
      },
    },
  },
  args: {
    trial: { eligible: true, offer: OFFER },
    isStarting: false,
    onStart: fn(),
  },
} satisfies Meta<typeof TrialOffer>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Starting: Story = {
  args: { isStarting: true },
};

export const YearlyMax: Story = {
  args: {
    trial: {
      eligible: true,
      offer: {
        ...OFFER,
        tier: "MAX",
        billing_cycle: "yearly",
        unit_amount: 192000,
        duration_days: 14,
      },
    },
  },
};

export const TeamInEuros: Story = {
  args: {
    trial: {
      eligible: true,
      offer: { ...OFFER, tier: "BUSINESS", currency: "eur", unit_amount: 4900 },
    },
  },
};

export const YenNoDecimals: Story = {
  args: {
    trial: {
      eligible: true,
      offer: { ...OFFER, tier: "BASIC", currency: "jpy", unit_amount: 1500 },
    },
  },
};
