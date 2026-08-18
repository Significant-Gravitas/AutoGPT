import { useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  useGetSubscriptionStatus,
  useUpdateSubscriptionTier,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { getTierLabel } from "./helpers";

export type SubscriptionStatus = SubscriptionStatusResponse;

const TIER_ORDER = ["NO_TIER", "BASIC", "PRO", "MAX", "BUSINESS", "ENTERPRISE"];

export function useSubscriptionTierSection() {
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const searchParams = useSearchParams();
  const subscriptionStatus = searchParams.get("subscription");
  const router = useRouter();
  const pathname = usePathname();
  const { toast } = useToast();
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
    query: { select: (data) => (data.status === 200 ? data.data : null) },
  });

  const fetchError = queryError ? "Failed to load subscription info" : null;

  const {
    mutateAsync: doUpdateTier,
    isPending,
    variables,
  } = useUpdateSubscriptionTier();

  useEffect(() => {
    if (subscriptionStatus === "success") {
      refetch();
      toast({
        title: "Subscription upgraded",
        description:
          "Your plan has been updated. It may take a moment to reflect.",
      });
    }
    // Strip ?subscription=success|cancelled from the URL so a page refresh
    // does not re-trigger side-effects, and so a second checkout in the same
    // session correctly fires the toast again.
    if (
      subscriptionStatus === "success" ||
      subscriptionStatus === "cancelled"
    ) {
      router.replace(pathname);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refetch and toast
    // are new references each render but are stable in practice; the effect must
    // only re-run when subscriptionStatus/pathname changes.
  }, [subscriptionStatus, refetch, toast, router, pathname]);

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
      if (result.status === 200 && result.data.url) {
        window.location.href = result.data.url;
        return;
      }
      await refetch();
      const currentIdx = subscription
        ? TIER_ORDER.indexOf(subscription.tier)
        : -1;
      const targetIdx = TIER_ORDER.indexOf(tier);
      const isDowngrade = targetIdx >= 0 && targetIdx < currentIdx;
      toast({
        title: "Subscription updated",
        description:
          tier === "NO_TIER"
            ? "Your subscription is cancelled at the end of your current billing period; no further charges."
            : isDowngrade
              ? `Your plan will be downgraded to ${getTierLabel(tier)} at the end of your current billing period; from then your saved card is billed at the new lower rate.`
              : `Upgraded to ${getTierLabel(tier)}. On the next invoice your saved card is charged for the upgrade proration plus the next month at the new rate.`,
      });
    } catch (e: unknown) {
      const msg =
        e instanceof Error ? e.message : "Failed to change subscription tier";
      setTierError(msg);
    }
  }

  function handleTierChange(
    targetTierKey: string,
    currentTier: string,
    onConfirmDowngrade: (tier: string) => void,
  ) {
    const currentIdx = TIER_ORDER.indexOf(currentTier);
    const targetIdx = TIER_ORDER.indexOf(targetTierKey);
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
      // "Stay on my current tier" is a same-tier POST: the backend collapses
      // cancel-pending into update-tier and releases any pending schedule.
      // success_url/cancel_url are unused in this branch (no Stripe Checkout
      // is created) but are sent to satisfy the request schema.
      await doUpdateTier({
        data: {
          tier: subscription.tier as SubscriptionTierRequestTier,
          success_url: `${window.location.origin}${window.location.pathname}`,
          cancel_url: `${window.location.origin}${window.location.pathname}`,
        },
      });
      await refetch();
      toast({
        title: "Pending subscription change cancelled.",
      });
    } catch (e: unknown) {
      const msg =
        e instanceof Error
          ? e.message
          : "Failed to cancel pending subscription change";
      setTierError(msg);
      toast({
        title: "Failed to cancel pending change",
        description: msg,
        variant: "destructive",
      });
      // Refetch on error so the UI reconciles if the server actually
      // succeeded (e.g. webhook delivered after our client-side error).
      // Swallow refetch errors — we already have the primary error for display.
      try {
        await refetch();
      } catch {
        // intentional
      }
    }
  }

  const pendingTier =
    isPending && variables?.data?.tier ? variables.data.tier : null;

  return {
    subscription: subscription ?? null,
    isLoading,
    error: fetchError,
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
  };
}
