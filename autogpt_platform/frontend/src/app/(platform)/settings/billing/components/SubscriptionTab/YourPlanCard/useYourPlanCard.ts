"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import {
  useGetSubscriptionStatus,
  useGetV1ManagePaymentMethods,
  useUpdateSubscriptionTier,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { GetV1ManagePaymentMethods200 } from "@/app/api/__generated__/models/getV1ManagePaymentMethods200";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";
import { toast } from "@/components/molecules/Toast/use-toast";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

import { TIER_ORDER, getTierLabel } from "../../../helpers";

// Team upgrade is contact-sales — open this page rather than firing a
// Stripe Checkout from the BUSINESS tier card. Replace once marketing
// publishes the canonical sales URL.
export const TEAM_UPGRADE_URL = "https://agpt.co/contact-sales";

export function useYourPlanCard() {
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const searchParams = useSearchParams();
  const subscriptionStatus = searchParams.get("subscription");
  const router = useRouter();
  const pathname = usePathname();

  const [tierError, setTierError] = useState<string | null>(null);
  const [pendingUpgradeTier, setPendingUpgradeTier] = useState<string | null>(
    null,
  );

  const {
    data: subscription,
    isLoading,
    error: queryError,
    refetch,
  } = useGetSubscriptionStatus({
    query: {
      select: (res) =>
        res.status === 200
          ? (res.data as SubscriptionStatusResponse)
          : undefined,
    },
  });

  const paymentPortal = useGetV1ManagePaymentMethods({
    query: {
      select: (res) => {
        const raw = res.data as GetV1ManagePaymentMethods200 | undefined;
        const url = raw && typeof raw === "object" ? raw.url : undefined;
        return typeof url === "string" ? url : undefined;
      },
    },
  });

  const fetchError = queryError ? "Failed to load subscription info." : null;

  const {
    mutateAsync: doUpdateTier,
    isPending,
    variables,
  } = useUpdateSubscriptionTier();

  // Mirror the old SubscriptionTierSection: surface ?subscription=success|cancelled
  // as a toast and strip the param so a refresh doesn't re-trigger the toast.
  useEffect(
    function handleStripeReturn() {
      if (subscriptionStatus === "success") {
        void refetch();
        toast({
          title: "Subscription upgraded",
          description:
            "Your plan has been updated. It may take a moment to reflect.",
        });
      }
      if (
        subscriptionStatus === "success" ||
        subscriptionStatus === "cancelled"
      ) {
        router.replace(pathname);
      }
    },
    // refetch + router are stable React Query / Next.js handles; including
    // them keeps the deps lint rule happy without re-running on every render.
    [subscriptionStatus, refetch, router, pathname],
  );

  async function changeTier(tier: string) {
    setTierError(null);
    try {
      const successUrl = `${window.location.origin}${window.location.pathname}?subscription=success`;
      const cancelUrl = `${window.location.origin}${window.location.pathname}?subscription=cancelled`;
      const result = await doUpdateTier({
        data: {
          tier: tier as SubscriptionTierRequestTier,
          success_url: successUrl,
          cancel_url: cancelUrl,
        },
      });

      if (result.status === 200) {
        const data = result.data as { url?: string } | undefined;
        if (data?.url) {
          window.location.href = data.url;
          return;
        }
      }

      await refetch();

      const currentIdx = subscription
        ? TIER_ORDER.indexOf(subscription.tier as (typeof TIER_ORDER)[number])
        : -1;
      const targetIdx = TIER_ORDER.indexOf(
        tier as (typeof TIER_ORDER)[number],
      );
      const isDowngrade = targetIdx >= 0 && targetIdx < currentIdx;

      toast({
        title: "Subscription updated",
        description:
          tier === "NO_TIER"
            ? "Your subscription is cancelled at the end of your current billing period; no further charges."
            : isDowngrade
              ? `Your plan will be downgraded to ${getTierLabel(tier)} at the end of your current billing period; from then your saved card is billed at the new lower rate.`
              : `Upgraded to ${getTierLabel(tier)}. On the next invoice your saved card is charged for the upgrade proration plus the next month at the new rate; matching credits land in your AutoGPT balance once Stripe confirms the charge.`,
      });
    } catch (e: unknown) {
      const msg =
        e instanceof Error ? e.message : "Failed to change subscription tier.";
      setTierError(msg);
    }
  }

  function handleTierChange(
    targetTierKey: string,
    currentTier: string,
    onConfirmDowngrade: (tier: string) => void,
  ) {
    if (targetTierKey === "BUSINESS") {
      // Team upgrade is contact-sales — sidestep Stripe entirely.
      window.open(TEAM_UPGRADE_URL, "_blank", "noopener,noreferrer");
      return;
    }
    const currentIdx = TIER_ORDER.indexOf(
      currentTier as (typeof TIER_ORDER)[number],
    );
    const targetIdx = TIER_ORDER.indexOf(
      targetTierKey as (typeof TIER_ORDER)[number],
    );
    if (targetIdx < currentIdx) {
      onConfirmDowngrade(targetTierKey);
      return;
    }
    setPendingUpgradeTier(targetTierKey);
  }

  async function confirmUpgrade() {
    if (!pendingUpgradeTier) return;
    const tier = pendingUpgradeTier;
    setPendingUpgradeTier(null);
    await changeTier(tier);
  }

  async function cancelPendingChange() {
    if (!subscription) return;
    setTierError(null);
    try {
      await doUpdateTier({
        data: {
          tier: subscription.tier as SubscriptionTierRequestTier,
          success_url: `${window.location.origin}${window.location.pathname}`,
          cancel_url: `${window.location.origin}${window.location.pathname}`,
        },
      });
      await refetch();
      toast({ title: "Pending subscription change cancelled." });
    } catch (e: unknown) {
      const msg =
        e instanceof Error
          ? e.message
          : "Failed to cancel pending subscription change.";
      setTierError(msg);
      toast({
        title: "Failed to cancel pending change",
        description: msg,
        variant: "destructive",
      });
      try {
        await refetch();
      } catch {
        // intentionally swallowed — primary error already surfaced.
      }
    }
  }

  // Mirrors the old hook's pendingTier — used to flag the *button* the user
  // just clicked (vs the subscription's pending_tier from the backend).
  const pendingTierOnButton =
    isPending && variables?.data?.tier ? variables.data.tier : null;

  return {
    subscription: subscription ?? null,
    isLoading,
    error: fetchError,
    tierError,
    isPending,
    pendingTierOnButton,
    pendingUpgradeTier,
    setPendingUpgradeTier,
    confirmUpgrade,
    isPaymentEnabled,
    portalUrl: paymentPortal.data ?? null,
    canManagePortal: Boolean(paymentPortal.data),
    onManage: () => {
      if (paymentPortal.data) window.location.href = paymentPortal.data;
    },
    changeTier,
    handleTierChange,
    cancelPendingChange,
  };
}
