import { apiRequest } from "./client";

/** `SubscriptionTier` in backend/data/credit.py; NO_TIER means no plan. */
export type PlanTier =
  | "NO_TIER"
  | "TRIAL"
  | "BASIC"
  | "PRO"
  | "MAX"
  | "BUSINESS"
  | "ENTERPRISE";

/** The fields the app reads from GET /api/credits/subscription. */
export interface Subscription {
  tier: PlanTier;
  /** In cents, for `billing_cycle`. */
  monthly_cost: number;
  billing_cycle: "monthly" | "yearly";
  /** Unix seconds; null without an active Stripe subscription. */
  current_period_end: number | null;
  pending_tier: PlanTier | null;
  pending_tier_effective_at: string | null;
  /** Set with `pending_tier` for a cycle change, even on the same tier. */
  pending_billing_cycle: "monthly" | "yearly" | null;
}

export interface TrialStatus {
  active: boolean;
  ends_at: string | null;
  cancel_at_period_end: boolean;
  allowance_used_percent: number | null;
}

export interface UsageWindow {
  percent_used: number;
  resets_at: string;
}

export interface ChatUsage {
  daily: UsageWindow | null;
  weekly: UsageWindow | null;
}

interface Options {
  signal?: AbortSignal;
}

export const billingKeys = {
  all: ["billing"] as const,
  checks: ["billing", "check"] as const,
  check: (userID: string) => ["billing", "check", userID] as const,
  subscription: ["billing", "subscription"] as const,
  trial: ["billing", "trial"] as const,
  usage: ["billing", "usage"] as const,
  credits: ["billing", "credits"] as const,
};

/** Whether this server sells plans at all (`ENABLE_PLATFORM_PAYMENT`). */
export function getPlatformConfig({ signal }: Options = {}) {
  return apiRequest<{ payment_enabled: boolean }>("/api/platform-config", {
    signal,
  });
}

export function getSubscription({ signal }: Options = {}) {
  return apiRequest<Subscription>("/api/credits/subscription", { signal });
}

export function getTrial() {
  return apiRequest<TrialStatus>("/api/credits/trial");
}

export function getChatUsage() {
  return apiRequest<ChatUsage>("/api/chat/usage");
}

/** The balance for workflow runs, in cents. */
export async function getCredits() {
  const { credits } = await apiRequest<{ credits: number }>("/api/credits");
  return credits;
}
