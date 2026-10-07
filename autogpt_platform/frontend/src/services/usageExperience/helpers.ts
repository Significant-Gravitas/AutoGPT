interface UsageWindow {
  percent_used: number;
  resets_at: string | Date;
}
interface Usage {
  tier?: string | null;
  daily?: UsageWindow | null;
  weekly?: UsageWindow | null;
}
interface Trial {
  status?: string | null;
  offer?: { tier?: string } | null;
  active?: boolean;
  converted?: boolean;
  allowance_used_percent?: number | null;
  ends_at?: string | Date | null;
  usage_policy?: string | null;
}

export function getUsageExperience(
  usage?: Usage | null,
  trial?: Trial | null,
  subscriptionTier?: string | null,
) {
  const tier = subscriptionTier ?? usage?.tier ?? null;
  const isActiveTrial =
    tier === "TRIAL" && trial?.active === true && !trial.converted;
  const isLifetimeTrial = isActiveTrial && trial?.usage_policy === "lifetime";
  const trialPercent = isActiveTrial
    ? (trial?.allowance_used_percent ?? null)
    : null;
  const trialEnded =
    tier === "NO_TIER" && trial && !trial.active && !trial.converted;
  const inactiveTrialStatus = trialEnded
    ? ["canceled", "cancelled"].includes(trial.status ?? "")
      ? "canceled"
      : ["past_due", "unpaid", "payment_failed"].includes(trial.status ?? "")
        ? "payment_failed"
        : ["expired", "incomplete_expired"].includes(trial.status ?? "") ||
            (trial.status === "trialing" &&
              !!trial.ends_at &&
              new Date(trial.ends_at).getTime() <= Date.now())
          ? "expired"
          : null
    : null;
  const exhausted = (trialPercent ?? 0) >= 100;
  const recurring =
    isLifetimeTrial || inactiveTrialStatus
      ? []
      : (["daily", "weekly"] as const).filter(
          (key) => (usage?.[key]?.percent_used ?? 0) >= 100,
        );
  const window = exhausted
    ? "trial"
    : (recurring.sort(
        (a, b) =>
          new Date(usage?.[b]?.resets_at ?? "").getTime() -
          new Date(usage?.[a]?.resets_at ?? "").getTime(),
      )[0] ?? null);
  const isTopTier =
    tier === "MAX" || tier === "BUSINESS" || tier === "ENTERPRISE";
  return {
    tier,
    isActiveTrial,
    isLifetimeTrial,
    freshProUsage: isActiveTrial && trial?.offer?.tier === "PRO",
    trialPercent,
    inactiveTrialStatus,
    trialEndsAt: isActiveTrial ? (trial?.ends_at ?? null) : null,
    blocked: window !== null || !!inactiveTrialStatus,
    window,
    resetsAt:
      window && window !== "trial"
        ? (usage?.[window]?.resets_at ?? null)
        : null,
    targetTier:
      isTopTier || !tier || inactiveTrialStatus === "payment_failed"
        ? null
        : tier === "PRO"
          ? ("MAX" as const)
          : ("PRO" as const),
    isTopTier,
  };
}

export type UsageExperience = ReturnType<typeof getUsageExperience>;

export function formatUsagePercent(value: number) {
  if (value > 0 && value < 1) return "<1%";
  if (value < 100 && value > 99) return "99%";
  return `${Math.min(100, Math.max(0, Math.round(value)))}%`;
}

export function formatUsageDate(value: string | Date) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  });
}
