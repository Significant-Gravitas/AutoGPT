"use client";

import { useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { ArrowSquareOutIcon } from "@phosphor-icons/react";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";

import {
  EASE_OUT,
  PLAN_TIERS,
  TIER_ORDER,
  formatRelativeMultiplier,
  formatShortDate,
  formatTierCost,
  getTierLabel,
} from "../../../helpers";
import { useYourPlanCard } from "./useYourPlanCard";

const PRICING_PAGE_URL = "https://agpt.co/pricing";

interface Props {
  index?: number;
}

export function YourPlanCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const {
    subscription,
    isLoading,
    error,
    tierError,
    isPending,
    pendingTierOnButton,
    pendingUpgradeTier,
    setPendingUpgradeTier,
    confirmUpgrade,
    isPaymentEnabled,
    canManagePortal,
    onManage,
    changeTier,
    handleTierChange,
    cancelPendingChange,
  } = useYourPlanCard();

  const [confirmDowngradeTo, setConfirmDowngradeTo] = useState<string | null>(
    null,
  );
  const [confirmReplacePendingTo, setConfirmReplacePendingTo] = useState<
    string | null
  >(null);

  if (isLoading) {
    return (
      <div className="flex w-full flex-col gap-3">
        <Skeleton className="h-6 w-32" />
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <Skeleton className="h-44 rounded-[18px]" />
          <Skeleton className="h-44 rounded-[18px]" />
          <Skeleton className="h-44 rounded-[18px]" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col gap-2 rounded-[18px] border border-red-200 bg-red-50 p-4">
        <Text variant="body-medium" as="span" className="text-red-700">
          {error}
        </Text>
      </div>
    );
  }

  if (!subscription) return null;

  const currentTier = subscription.tier;

  if (currentTier === "ENTERPRISE") {
    return (
      <motion.section
        initial={reduceMotion ? false : { opacity: 0, y: 12 }}
        animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
        transition={
          reduceMotion
            ? undefined
            : { duration: 0.32, ease: EASE_OUT, delay: 0.04 + index * 0.05 }
        }
        className="flex w-full flex-col gap-2"
      >
        <SectionHeader />
        <div className="rounded-[18px] border border-violet-300 bg-violet-50 p-5">
          <Text variant="large-medium" as="span" className="text-violet-800">
            Enterprise plan
          </Text>
          <Text variant="body" as="p" className="mt-1 text-zinc-700">
            Your Enterprise plan is managed by your administrator. Contact your
            account team for changes.
          </Text>
        </div>
      </motion.section>
    );
  }

  const pendingTierFromSubscription = subscription.pending_tier ?? null;
  const hasPendingChange =
    pendingTierFromSubscription !== null &&
    pendingTierFromSubscription !== currentTier;

  // Treat unpaid users as NO_TIER for picker comparisons even if the DB tier
  // defaults to PRO (rate-limit default per schema.prisma). Otherwise the
  // "current" highlight + Upgrade/Downgrade direction would be wrong for
  // brand-new users who haven't subscribed yet.
  const effectiveCurrentTier = subscription.has_active_stripe_subscription
    ? currentTier
    : "NO_TIER";

  function onTierButtonClick(targetTierKey: string) {
    // Pending change exists + clicked a *different* non-current tier → ask
    // before silently overwriting the scheduled change.
    if (
      hasPendingChange &&
      targetTierKey !== pendingTierFromSubscription &&
      targetTierKey !== currentTier
    ) {
      setConfirmReplacePendingTo(targetTierKey);
      return;
    }
    handleTierChange(targetTierKey, effectiveCurrentTier, setConfirmDowngradeTo);
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
    handleTierChange(tier, effectiveCurrentTier, setConfirmDowngradeTo);
  }

  const needsSubscription =
    isPaymentEnabled && !subscription.has_active_stripe_subscription;

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : { duration: 0.32, ease: EASE_OUT, delay: 0.04 + index * 0.05 }
      }
      className="flex w-full flex-col gap-3"
    >
      <SectionHeader />

      {needsSubscription ? (
        <div
          role="status"
          className="rounded-[14px] border border-violet-300 bg-violet-50 px-4 py-3"
        >
          <Text variant="body-medium" as="p" className="text-violet-900">
            Pick a plan to continue using AutoGPT.
          </Text>
          <Text variant="small" as="p" className="mt-1 text-violet-800">
            Your account doesn&apos;t have an active subscription. Choose a tier
            below to unlock AutoPilot and start running agents.
          </Text>
        </div>
      ) : null}

      {tierError ? (
        <div
          role="alert"
          className="rounded-[14px] border border-red-200 bg-red-50 px-3 py-2"
        >
          <Text variant="small" as="span" className="text-red-700">
            {tierError}
          </Text>
        </div>
      ) : null}

      {hasPendingChange && pendingTierFromSubscription ? (
        <PendingChangeBanner
          currentTier={currentTier}
          pendingTier={pendingTierFromSubscription}
          pendingEffectiveAt={subscription.pending_tier_effective_at ?? null}
          onKeepCurrent={() => void cancelPendingChange()}
          isBusy={isPending}
        />
      ) : null}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {PLAN_TIERS.map((tier) => {
          const isCurrent = effectiveCurrentTier === tier.key;
          const cost = subscription.tier_costs[tier.key] ?? 0;
          const currentIdx = TIER_ORDER.indexOf(
            effectiveCurrentTier as (typeof TIER_ORDER)[number],
          );
          const targetIdx = TIER_ORDER.indexOf(
            tier.key as (typeof TIER_ORDER)[number],
          );
          const isUpgrade = targetIdx > currentIdx;
          const isDowngrade = targetIdx < currentIdx;
          const isThisLoading = pendingTierOnButton === tier.key;
          const isScheduled =
            hasPendingChange && pendingTierFromSubscription === tier.key;
          const rateLimitLabel = formatRelativeMultiplier(
            tier.key,
            subscription.tier_multipliers ?? {},
          );

          const buttonLabel = isThisLoading
            ? "Updating..."
            : isScheduled
              ? "Scheduled"
              : tier.contactSales
                ? "Talk to sales"
                : isUpgrade
                  ? `Upgrade to ${tier.label}`
                  : isDowngrade
                    ? `Downgrade to ${tier.label}`
                    : `Switch to ${tier.label}`;

          return (
            <div
              key={tier.key}
              aria-current={isCurrent ? "true" : undefined}
              className={`flex flex-col gap-2 rounded-[18px] border p-5 shadow-[0_1px_2px_rgba(15,15,20,0.04)] ${
                isCurrent
                  ? "border-violet-500 bg-violet-50"
                  : "border-zinc-200 bg-white"
              }`}
            >
              <div className="flex items-center justify-between">
                <Text
                  variant="large-medium"
                  as="span"
                  className="text-textBlack"
                >
                  {tier.label}
                </Text>
                {isCurrent ? (
                  <Badge
                    variant="success"
                    size="small"
                    className="bg-violet-100 text-violet-800"
                  >
                    Current
                  </Badge>
                ) : null}
              </div>

              <Text variant="h4" as="span" className="text-textBlack">
                {formatTierCost(cost, tier.contactSales)}
              </Text>
              {rateLimitLabel ? (
                <Text variant="small" as="span" className="text-zinc-700">
                  {rateLimitLabel}
                </Text>
              ) : null}
              <Text variant="small" as="p" className="text-zinc-600">
                {tier.description}
              </Text>

              <div className="mt-auto pt-2">
                {!isCurrent && (isPaymentEnabled || tier.contactSales) ? (
                  <Button
                    className="w-full"
                    variant={isUpgrade || tier.contactSales ? "primary" : "outline"}
                    size="small"
                    disabled={isPending || isScheduled}
                    loading={isThisLoading}
                    onClick={() => onTierButtonClick(tier.key)}
                    rightIcon={
                      tier.contactSales ? (
                        <ArrowSquareOutIcon size={14} aria-hidden="true" />
                      ) : undefined
                    }
                  >
                    {buttonLabel}
                  </Button>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>

      {effectiveCurrentTier !== "NO_TIER" && isPaymentEnabled ? (
        <div className="flex flex-wrap items-center justify-between gap-3 px-1 pt-1">
          <Text variant="small" as="p" className="text-zinc-600">
            Subscription managed through Stripe. Upgrades take effect
            immediately. Downgrades take effect at the end of your billing
            period.
          </Text>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="small"
              onClick={onManage}
              disabled={!canManagePortal}
            >
              Manage subscription
            </Button>
            {!hasPendingChange ? (
              <Button
                variant="ghost"
                size="small"
                disabled={isPending}
                onClick={() => setConfirmDowngradeTo("NO_TIER")}
              >
                Cancel subscription
              </Button>
            ) : null}
          </div>
        </div>
      ) : null}

      <Dialog
        title="Confirm downgrade"
        controlled={{
          isOpen: !!confirmDowngradeTo,
          set: (open) => {
            if (!open) setConfirmDowngradeTo(null);
          },
        }}
      >
        <Dialog.Content>
          <Text variant="body" as="p" className="text-zinc-700">
            {confirmDowngradeTo === "NO_TIER"
              ? `Cancelling your subscription schedules it to end at the close of your current billing period${
                  subscription.current_period_end
                    ? ` on ${formatShortDate(
                        subscription.current_period_end * 1000,
                      )}`
                    : ""
                } — no charge today and no further charges to your card. You keep your current plan and existing credits until then.`
              : `Switching to ${getTierLabel(
                  confirmDowngradeTo ?? "",
                )} takes effect at the end of your current billing period${
                  subscription.current_period_end
                    ? ` on ${formatShortDate(
                        subscription.current_period_end * 1000,
                      )}`
                    : ""
                } — no charge today. You keep your current plan until then. From that date your saved card is billed at the ${getTierLabel(
                  confirmDowngradeTo ?? "",
                )} rate, and matching credits are added to your AutoGPT balance with each paid invoice.`}{" "}
            Are you sure?
          </Text>
          <Dialog.Footer>
            <Button
              variant="outline"
              size="small"
              onClick={() => setConfirmDowngradeTo(null)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              size="small"
              onClick={() => void confirmDowngrade()}
            >
              Confirm
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
          <Text variant="body" as="p" className="text-zinc-700">
            You have a pending change to{" "}
            {getTierLabel(pendingTierFromSubscription ?? "")}
            {subscription.pending_tier_effective_at
              ? ` scheduled for ${formatShortDate(
                  subscription.pending_tier_effective_at,
                )}`
              : ""}
            . Switching to {getTierLabel(confirmReplacePendingTo ?? "")} will
            replace it. Continue?
          </Text>
          <Dialog.Footer>
            <Button
              variant="outline"
              size="small"
              onClick={() => setConfirmReplacePendingTo(null)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              size="small"
              onClick={() => void confirmReplacePending()}
            >
              Replace pending change
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>

      <Dialog
        title="Confirm upgrade"
        controlled={{
          isOpen: !!pendingUpgradeTier,
          set: (open) => {
            if (!open) setPendingUpgradeTier(null);
          },
        }}
      >
        <Dialog.Content>
          <Text variant="body" as="p" className="text-zinc-700">
            {subscription.has_active_stripe_subscription
              ? `Your subscription is upgraded to ${getTierLabel(
                  pendingUpgradeTier ?? "",
                )} immediately. On your next invoice${
                  subscription.current_period_end
                    ? ` on ${formatShortDate(
                        subscription.current_period_end * 1000,
                      )}`
                    : ""
                }, your saved card is charged for the upgrade proration since today plus the next month at the new rate, with the unused portion of your current plan automatically deducted. Credits matching the paid amount are added to your AutoGPT balance once Stripe confirms the charge.`
              : `You'll be redirected to Stripe to enter payment details and start your ${getTierLabel(
                  pendingUpgradeTier ?? "",
                )} subscription. The first invoice's amount is added to your AutoGPT balance once Stripe confirms the charge.`}
          </Text>
          <Dialog.Footer>
            <Button
              variant="outline"
              size="small"
              onClick={() => setPendingUpgradeTier(null)}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              size="small"
              onClick={() => void confirmUpgrade()}
            >
              {subscription.has_active_stripe_subscription
                ? "Confirm upgrade"
                : "Continue to checkout"}
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </motion.section>
  );
}

function SectionHeader() {
  return (
    <div className="flex items-center gap-2 px-4">
      <Text variant="body-medium" as="span" className="text-textBlack">
        Your plan
      </Text>
      <a
        href={PRICING_PAGE_URL}
        target="_blank"
        rel="noopener noreferrer"
        aria-label="Compare plans on the AutoGPT pricing page"
        className="inline-flex items-center gap-1 text-zinc-500 hover:text-violet-700"
      >
        <Text variant="small" as="span">
          Compare plans
        </Text>
        <ArrowSquareOutIcon size={14} aria-hidden="true" />
      </a>
    </div>
  );
}

interface PendingChangeBannerProps {
  currentTier: string;
  pendingTier: string;
  pendingEffectiveAt: string | null;
  onKeepCurrent: () => void;
  isBusy: boolean;
}

function PendingChangeBanner({
  currentTier,
  pendingTier,
  pendingEffectiveAt,
  onKeepCurrent,
  isBusy,
}: PendingChangeBannerProps) {
  const isCancel = pendingTier === "NO_TIER";
  const dateLabel = pendingEffectiveAt
    ? formatShortDate(pendingEffectiveAt)
    : null;

  return (
    <div
      role="status"
      className="flex flex-wrap items-center justify-between gap-3 rounded-[14px] border border-amber-300 bg-amber-50 px-4 py-3"
    >
      <Text variant="small" as="p" className="text-amber-900">
        {isCancel
          ? `Your ${getTierLabel(currentTier)} subscription will end${
              dateLabel ? ` on ${dateLabel}` : " at the end of the current billing period"
            }.`
          : `Switching from ${getTierLabel(currentTier)} to ${getTierLabel(
              pendingTier,
            )}${dateLabel ? ` on ${dateLabel}` : " at the end of the current billing period"}.`}
      </Text>
      <Button
        variant="outline"
        size="small"
        disabled={isBusy}
        onClick={onKeepCurrent}
      >
        {isCancel ? "Resume subscription" : "Cancel downgrade"}
      </Button>
    </div>
  );
}
