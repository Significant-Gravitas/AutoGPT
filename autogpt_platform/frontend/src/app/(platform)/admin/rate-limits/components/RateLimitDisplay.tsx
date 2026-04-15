"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { UsageBar } from "../../components/UsageBar";

const TIERS = ["FREE", "PRO", "BUSINESS", "ENTERPRISE"] as const;
type Tier = (typeof TIERS)[number];

const TIER_MULTIPLIERS: Record<Tier, string> = {
  FREE: "1x base limits",
  PRO: "5x base limits",
  BUSINESS: "20x base limits",
  ENTERPRISE: "60x base limits",
};

const TIER_COLORS: Record<Tier, string> = {
  FREE: "bg-gray-100 text-gray-700",
  PRO: "bg-blue-100 text-blue-700",
  BUSINESS: "bg-purple-100 text-purple-700",
  ENTERPRISE: "bg-amber-100 text-amber-700",
};

interface Props {
  data: UserRateLimitResponse;
  onReset: (resetWeekly: boolean) => Promise<void>;
  onTierChange?: (newTier: string) => Promise<void>;
  /** Override the outer container classes (default: bordered card). */
  className?: string;
}

export function RateLimitDisplay({
  data,
  onReset,
  onTierChange,
  className,
}: Props) {
  const [isResetting, setIsResetting] = useState(false);
  const [resetWeekly, setResetWeekly] = useState(false);
  const [isChangingTier, setIsChangingTier] = useState(false);
  const { toast } = useToast();

  const currentTier = TIERS.includes(data.tier as Tier)
    ? (data.tier as Tier)
    : "FREE";

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

  async function handleTierChange(newTier: string) {
    if (newTier === currentTier || !onTierChange) return;
    if (
      !window.confirm(
        `Change tier from ${currentTier} to ${newTier}? This will change the user's rate limits.`,
      )
    )
      return;

    setIsChangingTier(true);
    try {
      await onTierChange(newTier);
      toast({
        title: "Tier updated",
        description: `Changed to ${newTier} (${TIER_MULTIPLIERS[newTier as Tier]}).`,
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to update tier.",
        variant: "destructive",
      });
    } finally {
      setIsChangingTier(false);
    }
  }

  const nothingToReset = resetWeekly
    ? data.daily_tokens_used === 0 && data.weekly_tokens_used === 0
    : data.daily_tokens_used === 0;

  return (
    <div className={className ?? "rounded-md border bg-white p-6"}>
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
          onChange={(e) => handleTierChange(e.target.value)}
          className="rounded-md border bg-white px-3 py-1.5 text-sm"
          disabled={isChangingTier || !onTierChange}
        >
          {TIERS.map((tier) => (
            <option key={tier} value={tier}>
              {tier} — {TIER_MULTIPLIERS[tier]}
            </option>
          ))}
        </select>
        {isChangingTier && (
          <span className="text-xs text-gray-500">Updating...</span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">Daily Usage</h3>
          <UsageBar
            used={data.daily_tokens_used}
            limit={data.daily_token_limit}
          />
        </div>
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">Weekly Usage</h3>
          <UsageBar
            used={data.weekly_tokens_used}
            limit={data.weekly_token_limit}
          />
        </div>
      </div>

      <div className="mt-6 flex items-center gap-3 border-t pt-4">
        <select
          aria-label="Reset scope"
          value={resetWeekly ? "both" : "daily"}
          onChange={(e) => setResetWeekly(e.target.value === "both")}
          className="rounded-md border bg-white px-3 py-1.5 text-sm"
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
