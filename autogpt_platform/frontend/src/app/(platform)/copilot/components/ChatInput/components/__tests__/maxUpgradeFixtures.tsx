import { getGetV2ListChatConnectionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getGetSubscriptionStatusMockHandler200 } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import { server } from "@/mocks/mock-server";
import { screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { ConnectionPicker } from "../ConnectionPicker/ConnectionPicker";

export const draft = "Compare the alternatives and keep my current draft.";

export function ComposerHarness() {
  return (
    <>
      <textarea aria-label="Message" />
      <ConnectionPicker />
    </>
  );
}

export async function openPicker() {
  await userEvent.click(
    await screen.findByRole("button", { name: /— change/ }),
  );
}

export async function openOffer() {
  await openPicker();
  await userEvent.click(
    await screen.findByRole("button", { name: "Upgrade to Max" }),
  );
  return screen.findByRole("dialog", { name: "Unlock Advanced with Max." });
}

export function deploymentOffer(
  overrides: Partial<AIConnectionOffer> = {},
): AIConnectionOffer {
  return {
    offer_id: "platform:deployment",
    provider_family: "autogpt",
    display_name: "AutoGPT Platform",
    auth_method: "deployment",
    credential_id: null,
    backed_by_label: "Your AutoGPT plan",
    description: "New chats are backed by your AutoGPT plan.",
    state: "ready",
    selectable: true,
    is_default: true,
    tiers: [
      {
        tier: "standard",
        label: "Balanced",
        selectable: true,
        display_model: "sonnet-server",
      },
      {
        tier: "advanced",
        label: "Advanced",
        selectable: false,
        display_model: "opus-server",
        lock_reason: "A Max plan or higher is required for Advanced.",
      },
    ],
    limitations: [],
    ...overrides,
  };
}

export function proSubscription(
  overrides: Partial<SubscriptionStatusResponse> = {},
): SubscriptionStatusResponse {
  return {
    tier: "PRO",
    monthly_cost: 5100,
    tier_costs: { PRO: 5100, MAX: 24700 },
    tier_costs_yearly: { PRO: 52020, MAX: 251940 },
    billing_cycle: "monthly",
    tier_multipliers: { PRO: 1, MAX: 4 },
    proration_credit_cents: 1000,
    has_active_stripe_subscription: true,
    ...overrides,
  };
}

export function mockMaxUpgrade(
  subscription = proSubscription(),
  offers = [deploymentOffer()],
) {
  server.use(
    getGetV2ListChatConnectionsMockHandler200({ offers }),
    getGetSubscriptionStatusMockHandler200(subscription),
  );
}
