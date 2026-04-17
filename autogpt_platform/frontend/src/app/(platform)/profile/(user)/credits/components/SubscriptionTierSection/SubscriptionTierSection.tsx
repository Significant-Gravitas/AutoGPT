"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
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
    description: "Base AutoPilot capacity with standard rate limits",
  },
  {
    key: "PRO",
    label: "Pro",
    multiplier: "5x",
    description: "5x AutoPilot capacity — run 5× more tasks per day/week",
  },
  {
    key: "BUSINESS",
    label: "Business",
    multiplier: "20x",
    description: "20x AutoPilot capacity — ideal for teams and heavy workloads",
  },
];

const TIER_ORDER = ["FREE", "PRO", "BUSINESS", "ENTERPRISE"];

function formatCost(cents: number, tierKey: string): string {
  if (tierKey === "FREE") return "Free";
  if (cents === 0) return "Pricing available soon";
  return `$${(cents / 100).toFixed(2)}/mo`;
}

export function SubscriptionTierSection() {
  const {
    subscription,
    isLoading,
    error,
    tierError,
    isPending,
    pendingTier,
    pendingUpgradeTier,
    setPendingUpgradeTier,
    confirmUpgrade,
    isPaymentEnabled,
    changeTier,
    handleTierChange,
  } = useSubscriptionTierSection();
  const [confirmDowngradeTo, setConfirmDowngradeTo] = useState<string | null>(
    null,
  );

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-6 w-48" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <Skeleton className="h-40 rounded-lg" />
          <Skeleton className="h-40 rounded-lg" />
          <Skeleton className="h-40 rounded-lg" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Subscription Plan</h3>
        <p
          role="alert"
          className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400"
        >
          {error}
        </p>
      </div>
    );
  }

  if (!subscription) return null;

  const currentTier = subscription.tier;

  if (currentTier === "ENTERPRISE") {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Subscription Plan</h3>
        <div className="rounded-lg border border-violet-500 bg-violet-50 p-4 dark:bg-violet-900/20">
          <p className="font-semibold text-violet-700 dark:text-violet-200">
            Enterprise Plan
          </p>
          <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
            Your Enterprise plan is managed by your administrator. Contact your
            account team for changes.
          </p>
        </div>
      </div>
    );
  }

  async function confirmDowngrade() {
    if (!confirmDowngradeTo) return;
    const tier = confirmDowngradeTo;
    setConfirmDowngradeTo(null);
    await changeTier(tier);
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium">Subscription Plan</h3>

      {tierError && (
        <p
          role="alert"
          className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400"
        >
          {tierError}
        </p>
      )}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {TIERS.map((tier) => {
          const isCurrent = currentTier === tier.key;
          const cost = subscription.tier_costs[tier.key] ?? 0;
          const currentIdx = TIER_ORDER.indexOf(currentTier);
          const targetIdx = TIER_ORDER.indexOf(tier.key);
          const isUpgrade = targetIdx > currentIdx;
          const isDowngrade = targetIdx < currentIdx;
          const isThisPending = pendingTier === tier.key;

          return (
            <div
              key={tier.key}
              aria-current={isCurrent ? "true" : undefined}
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

              <p className="mb-1 text-2xl font-bold">
                {formatCost(cost, tier.key)}
              </p>
              <p className="mb-1 text-sm font-medium text-neutral-600 dark:text-neutral-400">
                {tier.multiplier} rate limits
              </p>
              <p className="mb-4 text-sm text-neutral-500 dark:text-neutral-400">
                {tier.description}
              </p>

              {!isCurrent && isPaymentEnabled && (
                <Button
                  className="w-full"
                  variant={isUpgrade ? "default" : "outline"}
                  disabled={isPending}
                  onClick={() =>
                    handleTierChange(
                      tier.key,
                      currentTier,
                      setConfirmDowngradeTo,
                    )
                  }
                >
                  {isThisPending
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

      {currentTier !== "FREE" && isPaymentEnabled && (
        <p className="text-sm text-neutral-500">
          Your subscription is managed through Stripe. Upgrades and paid-tier
          changes take effect immediately; downgrades to Free are scheduled for
          the end of the current billing period.
        </p>
      )}

      <Dialog
        title="Confirm Downgrade"
        controlled={{
          isOpen: !!confirmDowngradeTo,
          set: (open) => {
            if (!open) setConfirmDowngradeTo(null);
          },
        }}
      >
        <Dialog.Content>
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            {confirmDowngradeTo === "FREE"
              ? "Downgrading to Free will schedule your subscription to cancel at the end of your current billing period. You keep your current plan until then."
              : `Switching to ${TIERS.find((t) => t.key === confirmDowngradeTo)?.label ?? confirmDowngradeTo} will take effect immediately.`}{" "}
            Are you sure?
          </p>
          <Dialog.Footer>
            <Button
              variant="outline"
              onClick={() => setConfirmDowngradeTo(null)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => void confirmDowngrade()}
            >
              Confirm Downgrade
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>

      <Dialog
        title="Confirm Upgrade"
        controlled={{
          isOpen: !!pendingUpgradeTier,
          set: (open) => {
            if (!open) setPendingUpgradeTier(null);
          },
        }}
      >
        <Dialog.Content>
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            {subscription &&
              subscription.proration_credit_cents > 0 &&
              `Your unused ${currentTier.charAt(0) + currentTier.slice(1).toLowerCase()} subscription ($${(subscription.proration_credit_cents / 100).toFixed(2)}) will be applied as a credit to your next Stripe invoice. `}
            You will be redirected to Stripe to complete your upgrade to{" "}
            {TIERS.find((t) => t.key === pendingUpgradeTier)?.label ??
              pendingUpgradeTier}
            .
          </p>
          <Dialog.Footer>
            <Button
              variant="outline"
              onClick={() => setPendingUpgradeTier(null)}
            >
              Cancel
            </Button>
            <Button onClick={() => void confirmUpgrade()}>
              Continue to Checkout
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}
