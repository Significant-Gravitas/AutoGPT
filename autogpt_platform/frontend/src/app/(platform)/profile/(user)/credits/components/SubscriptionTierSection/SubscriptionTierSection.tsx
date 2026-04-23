"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { useSubscriptionTierSection } from "./useSubscriptionTierSection";
import { PendingChangeBanner } from "./components/PendingChangeBanner/PendingChangeBanner";
import {
  TIERS,
  TIER_ORDER,
  formatCost,
  formatPendingDate,
  getTierLabel,
} from "./helpers";

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
    cancelPendingChange,
  } = useSubscriptionTierSection();
  const [confirmDowngradeTo, setConfirmDowngradeTo] = useState<string | null>(
    null,
  );
  const [confirmReplacePendingTo, setConfirmReplacePendingTo] = useState<
    string | null
  >(null);

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

  async function confirmReplacePending() {
    if (!confirmReplacePendingTo) return;
    const tier = confirmReplacePendingTo;
    setConfirmReplacePendingTo(null);
    handleTierChange(tier, currentTier, setConfirmDowngradeTo);
  }

  const pendingTierFromSubscription = subscription.pending_tier ?? null;
  const hasPendingChange =
    pendingTierFromSubscription !== null &&
    pendingTierFromSubscription !== currentTier;

  function onTierButtonClick(targetTierKey: string) {
    // If a pending change is queued and the user clicks a DIFFERENT non-current,
    // non-pending tier, surface a confirmation so they don't silently overwrite
    // their own scheduled change. The on-card button for the pending tier itself
    // is already disabled; the primary cancel path is the banner.
    if (
      hasPendingChange &&
      targetTierKey !== pendingTierFromSubscription &&
      targetTierKey !== currentTier
    ) {
      setConfirmReplacePendingTo(targetTierKey);
      return;
    }
    handleTierChange(targetTierKey, currentTier, setConfirmDowngradeTo);
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

      {hasPendingChange && pendingTierFromSubscription ? (
        <PendingChangeBanner
          currentTier={currentTier}
          pendingTier={pendingTierFromSubscription}
          pendingEffectiveAt={subscription.pending_tier_effective_at}
          onKeepCurrent={() => void cancelPendingChange()}
          isBusy={isPending}
        />
      ) : null}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {TIERS.map((tier) => {
          const isCurrent = currentTier === tier.key;
          const cost = subscription.tier_costs[tier.key] ?? 0;
          const currentIdx = TIER_ORDER.indexOf(currentTier);
          const targetIdx = TIER_ORDER.indexOf(tier.key);
          const isUpgrade = targetIdx > currentIdx;
          const isDowngrade = targetIdx < currentIdx;
          const isThisPending = pendingTier === tier.key;
          const isScheduledTier =
            hasPendingChange && pendingTierFromSubscription === tier.key;

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
                  disabled={isPending || isScheduledTier}
                  onClick={() => onTierButtonClick(tier.key)}
                >
                  {isThisPending
                    ? "Updating..."
                    : isScheduledTier
                      ? "Scheduled"
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
          Your subscription is managed through Stripe. Upgrades take effect
          immediately. Downgrades take effect at the end of your current billing
          period.
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
              : `Switching to ${TIERS.find((t) => t.key === confirmDowngradeTo)?.label ?? confirmDowngradeTo} will take effect at the end of your current billing period. You keep your current plan until then.`}{" "}
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
        title="Replace pending change?"
        controlled={{
          isOpen: !!confirmReplacePendingTo,
          set: (open) => {
            if (!open) setConfirmReplacePendingTo(null);
          },
        }}
      >
        <Dialog.Content>
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            You have a pending change to{" "}
            {getTierLabel(pendingTierFromSubscription ?? "")}
            {subscription.pending_tier_effective_at
              ? ` scheduled for ${formatPendingDate(subscription.pending_tier_effective_at)}`
              : ""}
            . Switching to {getTierLabel(confirmReplacePendingTo ?? "")} will
            replace it. Continue?
          </p>
          <Dialog.Footer>
            <Button
              variant="outline"
              onClick={() => setConfirmReplacePendingTo(null)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => void confirmReplacePending()}
            >
              Replace pending change
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
