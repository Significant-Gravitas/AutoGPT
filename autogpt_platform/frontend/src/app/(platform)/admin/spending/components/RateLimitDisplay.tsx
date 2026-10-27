"use client";

// TODO(tokens): no role for these colours yet, so they stay raw:
// bg-amber-100 #FEF3C7, bg-blue-100 #DBEAFE, bg-gray-100 #F3F4F6,
// bg-indigo-100 #E0E7FF, bg-purple-100 #EFE8FE, bg-red-100 #FDECEC,
// text-blue-700 #1D4ED8, text-gray-500 #6B7280, text-gray-700 #374151,
// text-indigo-700 #4338CA, text-purple-700 #5424AE.

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";
import { UsageBar } from "../../components/UsageBar";

const TIERS = [
  "NO_TIER",
  "BASIC",
  "PRO",
  "MAX",
  "BUSINESS",
  "ENTERPRISE",
] as const;
type Tier = (typeof TIERS)[number];

const TIER_COLORS: Record<Tier, string> = {
  NO_TIER: "bg-red-100 text-error-foreground",
  BASIC: "bg-gray-100 text-gray-700",
  PRO: "bg-blue-100 text-blue-700",
  MAX: "bg-indigo-100 text-indigo-700",
  BUSINESS: "bg-purple-100 text-purple-700",
  ENTERPRISE: "bg-amber-100 text-warning-foreground",
};

function tierLabel(tier: Tier): string {
  return tier === "NO_TIER" ? "no access (paywalled)" : "tier limits";
}

interface Props {
  data: UserRateLimitResponse;
  onReset: (resetWeekly: boolean) => Promise<void>;
  /** Override the outer container classes (default: bordered card). */
  className?: string;
}

export function RateLimitDisplay({ data, onReset, className }: Props) {
  const [isResetting, setIsResetting] = useState(false);
  const [resetWeekly, setResetWeekly] = useState(false);

  const currentTier = TIERS.includes(data.tier as Tier)
    ? (data.tier as Tier)
    : "NO_TIER";

  async function handleReset() {
    const msg = resetWeekly
      ? "Reset both daily and weekly usage counters to zero?"
      : "Reset daily usage counter to zero?";
    if (!window.confirm(msg)) return;

    setIsResetting(true);
    try {
      await onReset(resetWeekly);
    } finally {
      setIsResetting(false);
    }
  }

  const nothingToReset = resetWeekly
    ? data.daily_cost_used_microdollars === 0 &&
      data.weekly_cost_used_microdollars === 0
    : data.daily_cost_used_microdollars === 0;

  return (
    <div className={className ?? "rounded-md border bg-surface p-6"}>
      <div className="mb-4 flex items-start justify-between">
        <div>
          <h2 className="mb-1 text-lg font-semibold">
            Rate Limits for {data.user_email ?? data.user_id}
          </h2>
          {data.user_email && (
            <p className="text-xs text-gray-500">User ID: {data.user_id}</p>
          )}
        </div>
        <span
          className={`rounded-full px-3 py-1 text-xs font-medium ${TIER_COLORS[currentTier] ?? "bg-gray-100 text-gray-700"}`}
        >
          {currentTier}
        </span>
      </div>

      <div className="mb-4 flex items-center gap-3">
        <label className="text-sm font-medium text-gray-700">
          Subscription Tier
        </label>
        <select
          aria-label="Subscription tier"
          value={currentTier}
          className="rounded-md border bg-surface px-3 py-1.5 text-sm"
          disabled
        >
          {TIERS.map((tier) => (
            <option key={tier} value={tier}>
              {tier} — {tierLabel(tier)}
            </option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">Daily Spend</h3>
          <UsageBar
            used={data.daily_cost_used_microdollars}
            limit={data.daily_cost_limit_microdollars}
          />
        </div>
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">Weekly Spend</h3>
          <UsageBar
            used={data.weekly_cost_used_microdollars}
            limit={data.weekly_cost_limit_microdollars}
          />
        </div>
      </div>

      <div className="mt-6 flex items-center gap-3 border-t pt-4">
        <select
          aria-label="Reset scope"
          value={resetWeekly ? "both" : "daily"}
          onChange={(e) => setResetWeekly(e.target.value === "both")}
          className="rounded-md border bg-surface px-3 py-1.5 text-sm"
          disabled={isResetting}
        >
          <option value="daily">Reset daily only</option>
          <option value="both">Reset daily + weekly</option>
        </select>
        <Button
          variant="outline"
          onClick={handleReset}
          disabled={isResetting || nothingToReset}
        >
          {isResetting ? "Resetting..." : "Reset Usage"}
        </Button>
      </div>
    </div>
  );
}
