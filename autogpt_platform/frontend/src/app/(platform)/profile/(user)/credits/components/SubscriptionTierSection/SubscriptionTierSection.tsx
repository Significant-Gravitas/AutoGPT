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
  formatRelativeMultiplier,
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

  // Gate the "Pick a plan" banner on the DB tier rather than
  // has_active_stripe_subscription so a transient Stripe outage doesn't show
  // the banner to active subscribers. Same rationale as PaywallGate.
  const needsSubscription = isPaymentEnabled && subscription.tier === "NO_TIER";

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium">Subscription Plan</h3>

      {needsSubscription && (
        <div
          role="status"
          className="rounded-md border border-violet-300 bg-violet-50 px-4 py-3 text-sm text-violet-900 dark:border-violet-700 dark:bg-violet-900/20 dark:text-violet-200"
        >
          <p className="font-medium">Pick a plan to continue using AutoGPT.</p>
          <p className="mt-1">
            Your account doesn&apos;t have an active subscription. Choose a tier
            below to unlock AutoPilot and start running agents.
          </p>
        </div>
      )}

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
        {TIERS.filter(
          (tier) => subscription.tier_costs[tier.key] !== undefined,
        ).map((tier) => {
          const isCurrent = currentTier === tier.key;
          const cost = subscription.tier_costs[tier.key] ?? 0;
          const currentIdx = TIER_ORDER.indexOf(currentTier);
          const targetIdx = TIER_ORDER.indexOf(tier.key);
          const isUpgrade = targetIdx > currentIdx;
          const isDowngrade = targetIdx < currentIdx;
          const isThisPending = pendingTier === tier.key;
          const isScheduledTier =
            hasPendingChange && pendingTierFromSubscription === tier.key;
          const rateLimitLabel = formatRelativeMultiplier(
            tier.key,
            subscription.tier_multipliers ?? {},
          );

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
              {rateLimitLabel && (
                <p className="mb-1 text-sm font-medium text-neutral-600 dark:text-neutral-400">
                  {rateLimitLabel}
                </p>
              )}
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

      {currentTier !== "NO_TIER" && isPaymentEnabled && (
        <div className="flex items-center justify-between gap-3">
          <p className="text-sm text-neutral-500">
            Your subscription is managed through Stripe. Upgrades take effect
            immediately. Downgrades take effect at the end of your current
            billing period.
          </p>
          {!hasPendingChange && (
            <Button
              variant="ghost"
              className="shrink-0 text-sm text-neutral-600 hover:text-red-600 dark:text-neutral-400"
              disabled={isPending}
              onClick={() => setConfirmDowngradeTo("NO_TIER")}
            >
              Cancel subscription
            </Button>
          )}
        </div>
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
            {confirmDowngradeTo === "NO_TIER"
              ? `Cancelling your subscription schedules it to end at the close of your current billing period${subscription.current_period_end ? ` on ${formatPendingDate(new Date(subscription.current_period_end * 1000))}` : ""} — no charge today and no further charges to your card. You keep your current plan and existing credits until then.`
              : `Switching to ${getTierLabel(confirmDowngradeTo ?? "")} takes effect at the end of your current billing period${subscription.current_period_end ? ` on ${formatPendingDate(new Date(subscription.current_period_end * 1000))}` : ""} — no charge today. You keep your current plan until then. From that date your saved card is billed at the ${getTierLabel(confirmDowngradeTo ?? "")} rate.`}{" "}
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
            {subscription.has_active_stripe_subscription
              ? `Your subscription is upgraded to ${getTierLabel(pendingUpgradeTier ?? "")} immediately. On your next invoice${subscription.current_period_end ? ` on ${formatPendingDate(new Date(subscription.current_period_end * 1000))}` : ""}, your saved card is charged for the upgrade proration since today plus the next month at the new rate, with the unused portion of your current plan automatically deducted.`
              : `You'll be redirected to Stripe to enter payment details and start your ${getTierLabel(pendingUpgradeTier ?? "")} subscription.`}
          </p>
          <Dialog.Footer>
            <Button
              variant="outline"
              onClick={() => setPendingUpgradeTier(null)}
            >
              Cancel
            </Button>
            <Button onClick={() => void confirmUpgrade()}>
              {subscription.has_active_stripe_subscription
                ? "Confirm Upgrade"
                : "Continue to Checkout"}
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}
