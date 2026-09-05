import { getGetTrialsGetTrialStatusResponseMock200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";

export const trialOffer: TrialOfferResponse = {
  token: "a".repeat(64),
  version: "trial-v1",
  duration_days: 7,
  tier: "PRO",
  billing_cycle: "monthly",
  unit_amount: 2000,
  currency: "usd",
  onboarding_credit_amount: 300,
};

export function trialResponse(overrides: Partial<TrialStatusResponse> = {}) {
  return getGetTrialsGetTrialStatusResponseMock200({
    eligible: false,
    offer: trialOffer,
    status: "trialing",
    ends_at: new Date("2030-09-17T15:00:00Z"),
    cancel_at_period_end: false,
    allowance_used_percent: 42.4,
    active: true,
    converted: false,
    onboarding_credits_previously_received: false,
    ...overrides,
  });
}

export function setTrialUser(userID: string | null = "trial-user") {
  useAuthStore.setState({
    user: userID
      ? { id: userID, email: `${userID}@example.com`, user_metadata: {} }
      : null,
    isUserLoading: false,
    hasLoadedUser: true,
  });
}

export function deferredTrialResponse<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}
