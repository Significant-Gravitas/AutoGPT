import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { fn } from "storybook/test";
import { COUNTRIES } from "@/components/molecules/PlanCard/countries";
import { PLANS } from "@/components/molecules/PlanCard/plans";
import type { SubscriptionPlansProps } from "./helpers";
import { SubscriptionPlans } from "./SubscriptionPlans";

const trialOffer = {
  token: "storybook-offer",
  version: "trial-v1",
  duration_days: 7,
  tier: "PRO",
  billing_cycle: "monthly",
  unit_amount: 5000,
  currency: "usd",
  onboarding_credit_amount: 300,
} as const;

const meta = {
  title: "Organisms/SubscriptionPlans",
  component: SubscriptionPlans,
  parameters: { layout: "fullscreen" },
  args: {
    plans: PLANS.map((plan) => ({
      ...plan,
      highlighted: false,
      badge: plan.key === "TEAM" ? plan.badge : null,
    })),
    country: COUNTRIES[0],
    billing: "monthly",
    trialOffer,
    isUpdatingTier: false,
    selectedPlan: null,
    isStartingTrial: false,
    trialError: null,
    onBillingChange: fn(),
    onStartTrial: fn(),
    onSelectPlan: fn(),
  },
  render: function Render(args) {
    return <InteractivePlans {...args} />;
  },
} satisfies Meta<typeof SubscriptionPlans>;

export default meta;
type Story = StoryObj<typeof meta>;

function InteractivePlans(args: SubscriptionPlansProps) {
  const [billing, setBilling] = useState(args.billing);
  return (
    <div className="min-h-screen bg-gray-100 py-5">
      <SubscriptionPlans
        {...args}
        billing={billing}
        onBillingChange={(cycle) => {
          args.onBillingChange(cycle);
          setBilling(cycle);
        }}
      />
    </div>
  );
}

export const MonthlyTrial: Story = {};
export const Yearly: Story = { args: { billing: "yearly" } };
export const NoTrial: Story = { args: { trialOffer: null } };
export const StartingTrial: Story = { args: { isStartingTrial: true } };
export const PaidCheckout: Story = {
  args: { isUpdatingTier: true, selectedPlan: "MAX" },
};
export const HighlightedMax: Story = { args: { plans: PLANS } };
export const TrialError: Story = {
  args: { trialError: "Unable to start trial checkout. Please try again." },
};
export const YearlyYenTrial: Story = {
  args: {
    billing: "yearly",
    trialOffer: {
      ...trialOffer,
      duration_days: 14,
      billing_cycle: "yearly",
      currency: "jpy",
      unit_amount: 72000,
    },
  },
};
