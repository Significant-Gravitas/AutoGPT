interface UsageWindow {
  percent_used: number;
  resets_at: string | Date;
}
interface Usage {
  tier?: string | null;
  daily?: UsageWindow | null;
  weekly?: UsageWindow | null;
  state?: string;
  limiting_window?: "daily" | "weekly" | "trial" | null;
  resets_at?: string | Date | null;
}
interface Trial {
  status?: string | null;
  offer?: { tier?: string } | null;
  active?: boolean;
  cancel_at_period_end?: boolean;
  converted?: boolean;
  allowance_used_percent?: number | null;
  ends_at?: string | Date | null;
  usage_policy?: string | null;
  payment_due?: {
    amount_due: number;
    currency: string;
    payment_url: string;
  } | null;
}

function getTrialPayment(trial?: Trial | null) {
  const due = trial?.payment_due;
  if (!due) return null;
  return {
    ...due,
    plan: trial.offer?.tier ?? null,
    startsOn: trial.ends_at ?? null,
  };
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
  const clientWindow = exhausted
    ? "trial"
    : (recurring.sort(
        (a, b) =>
          new Date(usage?.[b]?.resets_at ?? "").getTime() -
          new Date(usage?.[a]?.resets_at ?? "").getTime(),
      )[0] ?? null);
  // When a turn can't start, the server names the limit that decides, down
  // to a trial's own daily or weekly cap that the client can't tell apart
  // from its allowance.
  const window =
    usage?.state === "exhausted" && usage.limiting_window
      ? usage.limiting_window
      : clientWindow;
  const isTopTier =
    tier === "MAX" || tier === "BUSINESS" || tier === "ENTERPRISE";
  // The server knows when a window allows nothing (no plan, or a trial that
  // is over): it reports no reset, and none is shown.
  const refills = !(usage?.state === "exhausted" && usage.resets_at === null);
  return {
    tier,
    isActiveTrial,
    isLifetimeTrial,
    freshProUsage:
      isActiveTrial &&
      !trial?.cancel_at_period_end &&
      trial?.offer?.tier === "PRO",
    isCancelPending: isActiveTrial && trial?.cancel_at_period_end === true,
    trialPercent,
    inactiveTrialStatus,
    payment:
      inactiveTrialStatus === "payment_failed" ? getTrialPayment(trial) : null,
    trialEndsAt: isActiveTrial ? (trial?.ends_at ?? null) : null,
    blocked: window !== null || !!inactiveTrialStatus,
    window,
    resetsAt:
      refills && window && window !== "trial"
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

export function getUsageStatus(percent: number) {
  if (percent >= 100) return "reached" as const;
  return percent >= 80 ? ("low" as const) : ("plenty" as const);
}

export function getFeaturedUsagePercent(
  experience: UsageExperience,
  usage?: Usage | null,
) {
  if (
    experience.isLifetimeTrial ||
    (experience.isActiveTrial && experience.window === "trial")
  )
    return experience.trialPercent ?? 0;
  const featured =
    experience.window === "weekly" || !usage?.daily
      ? usage?.weekly
      : usage.daily;
  return featured?.percent_used ?? 0;
}

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

export function formatTrialDate(value: string | Date | null) {
  if (!value) return null;
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return null;
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export function formatTrialStartDescription(value: string | Date | null) {
  const date = formatTrialDate(value);
  return `Pro starts on its own ${date ? `on ${date}` : "when your trial ends"}. Start it now to keep working today.`;
}

export function formatTrialStartTitle(value: string | Date | null) {
  const date = formatTrialDate(value);
  return date ? `Pro starts on ${date}` : "Pro starts when your trial ends";
}

export function formatResumeTime(value: string | Date | null) {
  if (!value) return null;
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return null;
  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);
  const nextWeek = new Date(today);
  nextWeek.setDate(nextWeek.getDate() + 7);
  const day =
    date.toDateString() === today.toDateString()
      ? "today"
      : date.toDateString() === tomorrow.toDateString()
        ? "tomorrow"
        : date > today && date < nextWeek
          ? date.toLocaleDateString(undefined, { weekday: "long" })
          : date.toLocaleDateString(undefined, {
              month: "short",
              day: "numeric",
              ...(date.getFullYear() !== today.getFullYear()
                ? { year: "numeric" as const }
                : {}),
            });
  return `${day} at ${date.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  })}`;
}

export function formatOfferPrice(price?: number, currency = "USD") {
  if (price === undefined || !Number.isFinite(price)) return null;
  const digits =
    new Intl.NumberFormat(undefined, {
      style: "currency",
      currency,
    }).resolvedOptions().maximumFractionDigits ?? 2;
  const amount = price / 10 ** digits;
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency,
    minimumFractionDigits: Number.isInteger(amount) ? 0 : digits,
    maximumFractionDigits: digits,
  }).format(amount);
}

export function getUsageOfferAction(
  experience: UsageExperience,
  offer: { price?: number; currency?: string },
) {
  if (experience.targetTier === "MAX") return "Upgrade to Max";
  if (experience.freshProUsage) {
    const amount = formatOfferPrice(offer.price, offer.currency);
    return `Start Pro now${amount ? ` · ${amount}` : ""}`;
  }
  if (experience.isCancelPending) return "Subscribe now";
  return experience.tier === "BASIC" ? "Upgrade to Pro" : "Start Pro";
}
