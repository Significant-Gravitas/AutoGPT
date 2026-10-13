"use client";

import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";
import { formatRelativeReset, formatShortDate } from "../../../helpers";

export interface UsageWindowView {
  label: string;
  percent: number;
  prefix: string;
  value: string;
}

export function useAutopilotUsageCard() {
  const { experience, usage, trial, isLoading, isError, retry } =
    useUsageExperience();
  const today: UsageWindowView | null = usage?.daily
    ? {
        label: "Today",
        percent: usage.daily.percent_used,
        ...formatRelativeReset(usage.daily.resets_at),
      }
    : null;
  const week: UsageWindowView | null = usage?.weekly
    ? {
        label: "This week",
        percent: usage.weekly.percent_used,
        ...formatRelativeReset(usage.weekly.resets_at),
      }
    : null;
  const showTrialWindow =
    experience.isLifetimeTrial ||
    (experience.isActiveTrial && experience.window === "trial");
  const trialWindow: UsageWindowView | null =
    showTrialWindow && experience.trialPercent != null
      ? {
          label: "Trial usage",
          percent: experience.trialPercent,
          prefix: "Trial ends",
          value: formatShortDate(trial?.ends_at),
        }
      : null;
  const featured = showTrialWindow
    ? trialWindow
    : experience.window === "weekly" || !today
      ? week
      : today;
  const secondary = showTrialWindow ? null : featured === week ? today : week;
  const percent = featured?.percent ?? 0;
  const status =
    percent >= 100
      ? { label: "Allowance reached", className: "text-orange-700" }
      : percent >= 80
        ? { label: "Running low", className: "text-amber-700" }
        : { label: "Plenty left", className: "text-zinc-500" };
  return {
    status,
    featured,
    secondary,
    today,
    week,
    isLoading,
    isError,
    retry,
    isTrial: experience.isActiveTrial,
    lifetime: experience.isLifetimeTrial,
    blocked: experience.blocked,
    inactive: experience.tier === "NO_TIER",
    paymentFailed:
      experience.tier === "NO_TIER" &&
      !trial?.active &&
      !trial?.converted &&
      ["past_due", "unpaid"].includes(trial?.status ?? ""),
    hasUsage: Boolean(featured || secondary),
    unlimited: Boolean(
      usage &&
        !experience.isActiveTrial &&
        usage.daily === null &&
        usage.weekly === null,
    ),
  };
}
