"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useSubscriptionTierSection } from "./useSubscriptionTierSection";

type TierInfo = {
  key: string;
  label: string;
  multiplier: string;
  description: string;
};

const TIERS: TierInfo[] = [
  {
    key: "FREE",
    label: "Free",
    multiplier: "1x",
    description: "Base rate limits",
  },
  {
    key: "PRO",
    label: "Pro",
    multiplier: "5x",
    description: "5x more AutoPilot capacity",
  },
  {
    key: "BUSINESS",
    label: "Business",
    multiplier: "20x",
    description: "20x more AutoPilot capacity",
  },
];

function formatCost(cents: number): string {
  if (cents === 0) return "Free";
  return `$${(cents / 100).toFixed(2)}/mo`;
}

export function SubscriptionTierSection() {
  const { subscription, isLoading, error, isPending, changeTier } =
    useSubscriptionTierSection();
  const [tierError, setTierError] = useState<string | null>(null);

  if (isLoading) return null;

  if (error) {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Subscription Plan</h3>
        <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </p>
      </div>
    );
  }

  if (!subscription) return null;

  async function handleTierChange(tierKey: string) {
    setTierError(null);
    const err = await changeTier(tierKey);
    if (err) setTierError(err);
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium">Subscription Plan</h3>

      {tierError && (
        <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
          {tierError}
        </p>
      )}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {TIERS.map((tier) => {
          const isCurrent = subscription.tier === tier.key;
          const cost = subscription.tier_costs[tier.key] ?? 0;
          const currentTierOrder = ["FREE", "PRO", "BUSINESS", "ENTERPRISE"];
          const currentIdx = currentTierOrder.indexOf(subscription.tier);
          const targetIdx = currentTierOrder.indexOf(tier.key);
          const isUpgrade = targetIdx > currentIdx;
          const isDowngrade = targetIdx < currentIdx;

          return (
            <div
              key={tier.key}
              className={`rounded-lg border p-4 ${
                isCurrent
                  ? "border-violet-500 bg-violet-50 dark:bg-violet-900/20"
                  : "border-neutral-200 dark:border-neutral-700"
              }`}
            >
              <div className="mb-2 flex items-center justify-between">
                <span className="font-semibold">{tier.label}</span>
                {isCurrent && (
                  <span className="rounded-full bg-violet-100 px-2 py-0.5 text-xs font-medium text-violet-700 dark:bg-violet-800 dark:text-violet-200">
                    Current
                  </span>
                )}
              </div>

              <p className="mb-1 text-2xl font-bold">{formatCost(cost)}</p>
              <p className="mb-1 text-sm font-medium text-neutral-600 dark:text-neutral-400">
                {tier.multiplier} rate limits
              </p>
              <p className="mb-4 text-sm text-neutral-500 dark:text-neutral-400">
                {tier.description}
              </p>

              {!isCurrent && (
                <Button
                  className="w-full"
                  variant={isUpgrade ? "default" : "outline"}
                  disabled={isPending}
                  onClick={() => handleTierChange(tier.key)}
                >
                  {isPending
                    ? "Updating..."
                    : isUpgrade
                      ? `Upgrade to ${tier.label}`
                      : isDowngrade
                        ? `Downgrade to ${tier.label}`
                        : `Switch to ${tier.label}`}
                </Button>
              )}
            </div>
          );
        })}
      </div>

      {subscription.tier !== "FREE" && (
        <p className="text-sm text-neutral-500">
          Your subscription is managed through Stripe. Changes take effect
          immediately.
        </p>
      )}
    </div>
  );
}
