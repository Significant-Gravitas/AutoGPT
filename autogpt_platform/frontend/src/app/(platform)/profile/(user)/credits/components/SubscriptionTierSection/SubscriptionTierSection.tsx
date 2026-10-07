"use client";
import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
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
        <Text variant="h5" as="h3">
          Subscription Plan
        </Text>
        <Text
          variant="body"
          role="alert"
          className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-red-700"
        >
          {error}
        </Text>
      </div>
    );
  }

  if (!subscription) return null;

  const currentTier = subscription.tier;

  if (currentTier === "ENTERPRISE") {
    return (
      <div className="space-y-4">
        <Text variant="h5" as="h3">
          Subscription Plan
        </Text>
        <div className="rounded-lg border border-purple-500 bg-purple-50 p-4">
          <Text variant="large-semibold" className="text-purple-700">
            Enterprise Plan
          </Text>
          <Text variant="body" tone="secondary" className="mt-1">
            Your Enterprise plan is managed by your administrator. Contact your
            account team for changes.
          </Text>
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
      <Text variant="h5" as="h3">
        Subscription Plan
      </Text>

      {needsSubscription && (
        <div
          role="status"
          className="rounded-md border border-purple-300 bg-purple-50 px-4 py-3"
        >
          <Text variant="body-medium" className="text-purple-900">
            Pick a plan to continue using AutoGPT.
          </Text>
          <Text variant="body" className="mt-1 text-purple-900">
            Your account doesn&apos;t have an active subscription. Choose a tier
            below to start working with experts and running agents.
          </Text>
        </div>
      )}

      {tierError && (
        <Text
          variant="body"
          role="alert"
          className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-red-700"
        >
          {tierError}
        </Text>
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
                isCurrent ? "border-purple-500 bg-purple-50" : "border-zinc-200"
              }`}
            >
              <div className="mb-2 flex items-center justify-between">
                <Text variant="large-semibold" as="span">
                  {tier.label}
                </Text>
                {isCurrent && (
                  <Text
                    variant="small-medium"
                    as="span"
                    className="rounded-full bg-purple-100 px-2 py-0.5 text-purple-700"
                  >
                    Current
                  </Text>
                )}
              </div>

              <Text variant="h4" as="p" className="mb-1">
                {formatCost(cost, tier.key)}
              </Text>
              {rateLimitLabel && (
                <Text variant="body-medium" tone="secondary" className="mb-1">
                  {rateLimitLabel}
                </Text>
              )}
              <Text variant="body" tone="muted" className="mb-4">
                {tier.description}
              </Text>

              {!isCurrent && isPaymentEnabled && (
                <Button
                  size="md"
                  className="w-full"
                  variant={isUpgrade ? "primary" : "secondary"}
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
          <Text variant="body" tone="muted">
            Your subscription is managed through Stripe. Upgrades take effect
            immediately. Downgrades take effect at the end of your current
            billing period.
          </Text>
          {!hasPendingChange && (
            <Button
              variant="ghost"
              size="md"
              className="shrink-0 text-zinc-600 hover:text-red-600"
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
          <Text variant="body" tone="secondary">
            {confirmDowngradeTo === "NO_TIER"
              ? `Cancelling your subscription schedules it to end at the close of your current billing period${subscription.current_period_end ? ` on ${formatPendingDate(new Date(subscription.current_period_end * 1000))}` : ""} — no charge today and no further charges to your card. You keep your current plan and existing credits until then.`
              : `Switching to ${getTierLabel(confirmDowngradeTo ?? "")} takes effect at the end of your current billing period${subscription.current_period_end ? ` on ${formatPendingDate(new Date(subscription.current_period_end * 1000))}` : ""} — no charge today. You keep your current plan until then. From that date your saved card is billed at the ${getTierLabel(confirmDowngradeTo ?? "")} rate.`}{" "}
            Are you sure?
          </Text>
          <Dialog.Footer>
            <Button
              variant="secondary"
              size="md"
              onClick={() => setConfirmDowngradeTo(null)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              size="md"
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
          <Text variant="body" tone="secondary">
            You have a pending change to{" "}
            {getTierLabel(pendingTierFromSubscription ?? "")}
            {subscription.pending_tier_effective_at
              ? ` scheduled for ${formatPendingDate(subscription.pending_tier_effective_at)}`
              : ""}
            . Switching to {getTierLabel(confirmReplacePendingTo ?? "")} will
            replace it. Continue?
          </Text>
          <Dialog.Footer>
            <Button
              variant="secondary"
              size="md"
              onClick={() => setConfirmReplacePendingTo(null)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              size="md"
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
          <Text variant="body" tone="secondary">
            {subscription.has_active_stripe_subscription
              ? `Your subscription is upgraded to ${getTierLabel(pendingUpgradeTier ?? "")} immediately. On your next invoice${subscription.current_period_end ? ` on ${formatPendingDate(new Date(subscription.current_period_end * 1000))}` : ""}, your saved card is charged for the upgrade proration since today plus the next month at the new rate, with the unused portion of your current plan automatically deducted.`
              : `You'll be redirected to Stripe to enter payment details and start your ${getTierLabel(pendingUpgradeTier ?? "")} subscription.`}
          </Text>
          <Dialog.Footer>
            <Button
              variant="secondary"
              size="md"
              onClick={() => setPendingUpgradeTier(null)}
            >
              Cancel
            </Button>
            <Button size="md" onClick={() => void confirmUpgrade()}>
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
