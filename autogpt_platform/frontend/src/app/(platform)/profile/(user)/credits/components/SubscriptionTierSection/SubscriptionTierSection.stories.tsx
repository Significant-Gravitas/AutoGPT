import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import { SubscriptionTierSection } from "./SubscriptionTierSection";
import { PendingChangeBanner } from "./components/PendingChangeBanner/PendingChangeBanner";

// The tier buttons only show with billing on: build Storybook with
// NEXT_PUBLIC_FORCE_FLAG_ENABLE_PLATFORM_PAYMENT=true to see them.
const SUBSCRIPTION_URL = "*/api/credits/subscription";

function subscription(overrides: Record<string, unknown> = {}) {
  return {
    tier: "BASIC",
    monthly_cost: 0,
    tier_costs: { BASIC: 0, PRO: 1999, MAX: 32000, ENTERPRISE: 0 },
    tier_multipliers: { BASIC: 1, PRO: 5, MAX: 20, BUSINESS: 60 },
    proration_credit_cents: 0,
    pending_tier: null,
    pending_tier_effective_at: null,
    has_active_stripe_subscription: false,
    ...overrides,
  };
}

function handlers(data: Record<string, unknown>) {
  return [
    http.get(SUBSCRIPTION_URL, () => HttpResponse.json(data)),
    // A tier change never answers, so a click shows the busy state.
    http.post(SUBSCRIPTION_URL, async () => {
      await delay("infinite");
      return HttpResponse.json({});
    }),
  ];
}

const meta: Meta<typeof SubscriptionTierSection> = {
  title: "Billing/SubscriptionTierSection",
  component: SubscriptionTierSection,
  parameters: {
    layout: "padded",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  parameters: { msw: { handlers: handlers(subscription()) } },
};

export const ProActive: Story = {
  parameters: {
    msw: {
      handlers: handlers(
        subscription({
          tier: "PRO",
          monthly_cost: 1999,
          has_active_stripe_subscription: true,
        }),
      ),
    },
  },
};

export const MaxWithPendingDowngrade: Story = {
  parameters: {
    msw: {
      handlers: handlers(
        subscription({
          tier: "MAX",
          monthly_cost: 32000,
          has_active_stripe_subscription: true,
          pending_tier: "PRO",
          pending_tier_effective_at: "2026-02-15T00:00:00Z",
        }),
      ),
    },
  },
};

export const ProWithPendingCancellation: Story = {
  parameters: {
    msw: {
      handlers: handlers(
        subscription({
          tier: "PRO",
          monthly_cost: 1999,
          has_active_stripe_subscription: true,
          pending_tier: "NO_TIER",
          pending_tier_effective_at: "2026-02-15T00:00:00Z",
        }),
      ),
    },
  },
};

export const BannerKeepCurrent: StoryObj<typeof PendingChangeBanner> = {
  render: () => (
    <PendingChangeBanner
      currentTier="MAX"
      pendingTier="PRO"
      pendingEffectiveAt="2026-02-15T00:00:00Z"
      onKeepCurrent={() => {}}
      isBusy={false}
    />
  ),
};

export const BannerCancelling: StoryObj<typeof PendingChangeBanner> = {
  render: () => (
    <PendingChangeBanner
      currentTier="PRO"
      pendingTier="NO_TIER"
      pendingEffectiveAt="2026-02-15T00:00:00Z"
      onKeepCurrent={() => {}}
      isBusy
    />
  ),
};
