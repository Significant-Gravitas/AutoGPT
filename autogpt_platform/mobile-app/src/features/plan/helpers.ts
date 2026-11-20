// No native imports, so tests can load this without a phone.
import type {
  ChatUsage,
  PlanTier,
  Subscription,
  TrialStatus,
} from "../../lib/api/billing";

/** The web's names (frontend/src/services/billing/subscription-plan.ts). */
const PLAN_NAMES: Record<PlanTier, string> = {
  NO_TIER: "No plan",
  TRIAL: "Free trial",
  BASIC: "Basic",
  PRO: "Pro",
  MAX: "Max",
  BUSINESS: "Team",
  ENTERPRISE: "Enterprise",
};

export type BadgeTone = "active" | "trial" | "ending";

export interface PlanSummary {
  name: string;
  badge: { label: string; tone: BadgeTone };
  headline: string | null;
  detail: string | null;
}

export interface Meter {
  label: string;
  note: string;
  percent: number;
}

export function planName(tier: PlanTier) {
  return PLAN_NAMES[tier];
}

/** What the account is on and when money next moves; nothing to act on. */
export function planSummary(
  subscription: Subscription,
  trial: TrialStatus | undefined,
): PlanSummary {
  const name = planName(subscription.tier);
  if (subscription.tier === "NO_TIER") {
    return { name, badge: { label: "Inactive", tone: "ending" }, headline: null, detail: null };
  }
  if (subscription.tier === "TRIAL") return trialSummary(name, trial);
  const cost = costLine(subscription);
  const endsAt = subscription.pending_tier_effective_at;
  if (subscription.pending_tier === "NO_TIER" && endsAt) {
    return {
      name,
      badge: { label: `Ends ${shortDate(endsAt)}`, tone: "ending" },
      headline: `Your plan ends ${longDate(endsAt)}`,
      detail: "Nothing more will be charged.",
    };
  }
  const renews = subscription.current_period_end
    ? new Date(subscription.current_period_end * 1000)
    : null;
  return {
    name,
    badge: { label: "Active", tone: "active" },
    headline: changeLine(subscription) ?? (renews ? `Renews ${longDate(renews)}` : null),
    detail: cost,
  };
}

function trialSummary(name: string, trial: TrialStatus | undefined): PlanSummary {
  const endsAt = trial?.ends_at ?? null;
  if (trial?.cancel_at_period_end && endsAt) {
    return {
      name,
      badge: { label: `Ends ${shortDate(endsAt)}`, tone: "ending" },
      headline: "Your trial is cancelled",
      detail: `You keep access until ${longDate(endsAt)}. Nothing will be charged.`,
    };
  }
  return {
    name,
    badge: { label: "Trial", tone: "trial" },
    headline: endsAt ? `Your trial ends ${longDate(endsAt)}` : null,
    detail: null,
  };
}

function changeLine(subscription: Subscription) {
  const { pending_tier: next, pending_tier_effective_at: at } = subscription;
  if (!next || !at) return null;
  if (next !== subscription.tier) return `Changes to ${planName(next)} on ${longDate(at)}`;
  const cycle = subscription.pending_billing_cycle;
  if (!cycle || cycle === subscription.billing_cycle) return null;
  return `Switches to ${cycle} billing on ${longDate(at)}`;
}

function costLine({ monthly_cost: cents, billing_cycle: cycle }: Subscription) {
  if (cents <= 0) return null;
  return `${formatMoney(cents)} a ${cycle === "yearly" ? "year" : "month"}.`;
}

/** The usage meters, in the order the web shows them. */
export function usageMeters(
  usage: ChatUsage | undefined,
  trial: TrialStatus | undefined,
): Meter[] {
  const meters: Meter[] = [];
  const trialPercent = trial?.active ? trial.allowance_used_percent : null;
  if (trialPercent != null) {
    meters.push({ label: "Trial allowance", note: usedNote(trialPercent), percent: trialPercent });
  }
  if (usage?.weekly) {
    meters.push({ label: "This week", note: usedNote(usage.weekly.percent_used), percent: usage.weekly.percent_used });
  }
  if (usage?.daily) {
    meters.push({
      label: "Today",
      note: `Resets at ${shortTime(usage.daily.resets_at)}`,
      percent: usage.daily.percent_used,
    });
  }
  return meters;
}

function usedNote(percent: number) {
  return `${Math.round(clampPercent(percent))}% used`;
}

export function clampPercent(percent: number) {
  return Math.min(100, Math.max(0, percent));
}

export function formatMoney(cents: number) {
  return `$${(cents / 100).toFixed(2)}`;
}

function longDate(value: string | Date) {
  return new Date(value).toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

function shortDate(value: string | Date) {
  return new Date(value).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function shortTime(value: string) {
  return new Date(value).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
}
