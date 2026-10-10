import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import { fireEvent, screen } from "@/tests/integrations/test-utils";
import { trialOffer, trialResponse } from "@/tests/integrations/trial-fixtures";
import { http, HttpResponse } from "msw";
import { vi } from "vitest";

export const cancelPending = {
  cancel_at_period_end: true,
  cancel_keeps_access: true,
};

export const trialSubscription: SubscriptionStatusResponse = {
  tier: "TRIAL",
  monthly_cost: 0,
  proration_credit_cents: 0,
  has_active_stripe_subscription: true,
  tier_costs: { PRO: 5000, MAX: 32000 },
  tier_costs_yearly: { PRO: 51000, MAX: 326400 },
  tier_multipliers: { PRO: 1, MAX: 8.5 },
};

export const yearlyOffer: TrialOfferResponse = {
  ...trialOffer,
  billing_cycle: "yearly",
  unit_amount: 51000,
};

export function returnURLs(plan: string, cycle: string) {
  const page = `${window.location.origin}${window.location.pathname}`;
  return {
    success_url: `${page}?subscription=success&session_id={CHECKOUT_SESSION_ID}&plan=${plan}&cycle=${cycle}`,
    cancel_url: `${page}?subscription=cancelled`,
  };
}

export function mockBilling(trial = trialResponse(cancelPending)) {
  const state = { trial, subscription: trialSubscription };
  const hits = { trial: 0, subscription: 0, invoices: 0 };
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(() => {
      hits.trial += 1;
      return state.trial;
    }),
    http.get("*/api/credits/subscription", () => {
      hits.subscription += 1;
      return HttpResponse.json(state.subscription);
    }),
    http.get("*/api/credits/invoices", () => {
      hits.invoices += 1;
      return HttpResponse.json([]);
    }),
  );
  return { state, hits };
}

export function mockPricePerUser() {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(() =>
      trialResponse({
        ...cancelPending,
        offer: {
          ...trialOffer,
          unit_amount:
            useAuthStore.getState().user?.id === "user-b" ? 3000 : 2000,
        },
      }),
    ),
  );
}

export function mockPlanRequest(respond: () => Response) {
  const body = vi.fn();
  server.use(
    http.post("*/api/credits/subscription", async ({ request }) => {
      body(await request.json());
      return respond();
    }),
  );
  return body;
}

export async function openConfirmation() {
  fireEvent.click(
    await screen.findByRole("button", { name: "Subscribe to Pro" }),
  );
  return screen.findByRole("dialog", { name: "Start Pro today?" });
}
