import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import { expect, userEvent, within } from "storybook/test";
import { SubscriptionTierSection } from "./SubscriptionTierSection";
import { PendingChangeBanner } from "./components/PendingChangeBanner/PendingChangeBanner";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";

// The tier buttons only show with billing on, which the server flags answer:
// the flags request is mocked with billing on, and the auth store marked as
// loaded so the flags request is made.
const SUBSCRIPTION_URL = "*/api/credits/subscription";
const FLAGS_URL = "*/api/feature-flags";
// A fixed date that stays in the future, so a pending change reads as upcoming.
const PENDING_EFFECTIVE_AT = "2099-02-15T00:00:00Z";

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

const flagsHandler = http.get(FLAGS_URL, () =>
  HttpResponse.json({
    flags: { "enable-platform-payment": true },
    configured: true,
  }),
);

function handlers(data: Record<string, unknown>) {
  return [
    flagsHandler,
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
  decorators: [
    (Story) => {
      useAuthStore.setState({ isUserLoading: false });
      return <Story />;
    },
  ],
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
          pending_tier_effective_at: PENDING_EFFECTIVE_AT,
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
          pending_tier_effective_at: PENDING_EFFECTIVE_AT,
        }),
      ),
    },
  },
};

export const LoadError: Story = {
  parameters: {
    msw: {
      handlers: [
        flagsHandler,
        http.get(SUBSCRIPTION_URL, () =>
          HttpResponse.json({ detail: "boom" }, { status: 500 }),
        ),
      ],
    },
  },
};

export const Enterprise: Story = {
  parameters: {
    msw: { handlers: handlers(subscription({ tier: "ENTERPRISE" })) },
  },
};

export const NoTier: Story = {
  parameters: {
    msw: { handlers: handlers(subscription({ tier: "NO_TIER" })) },
  },
};

export const TierChangeError: Story = {
  parameters: {
    msw: {
      handlers: [
        flagsHandler,
        http.get(SUBSCRIPTION_URL, () => HttpResponse.json(subscription())),
        http.post(SUBSCRIPTION_URL, () =>
          HttpResponse.json(
            { detail: "Payment method declined" },
            { status: 500 },
          ),
        ),
      ],
    },
  },
  // The upgrade asks for confirmation first; confirming sends the POST that
  // fails, and the section shows the error.
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const page = within(canvasElement.ownerDocument.body);
    await userEvent.click(
      await canvas.findByRole("button", { name: /upgrade to pro/i }),
    );
    await userEvent.click(
      await page.findByRole("button", { name: /continue to checkout/i }),
    );
    await expect(await canvas.findByRole("alert")).toBeVisible();
  },
};

export const BannerKeepCurrent: StoryObj<typeof PendingChangeBanner> = {
  render: () => (
    <PendingChangeBanner
      currentTier="MAX"
      pendingTier="PRO"
      pendingEffectiveAt={PENDING_EFFECTIVE_AT}
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
      pendingEffectiveAt={PENDING_EFFECTIVE_AT}
      onKeepCurrent={() => {}}
      isBusy
    />
  ),
};
