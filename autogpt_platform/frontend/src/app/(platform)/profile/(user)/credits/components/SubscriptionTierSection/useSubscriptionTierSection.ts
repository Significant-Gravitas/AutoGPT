import {
  useGetSubscriptionStatus,
  useUpdateSubscriptionTier,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";

export type SubscriptionStatus = SubscriptionStatusResponse;

export function useSubscriptionTierSection() {
  const {
    data: subscription,
    isLoading,
    error: queryError,
    refetch,
  } = useGetSubscriptionStatus({
    query: { select: (data) => (data.status === 200 ? data.data : null) },
  });

  const error = queryError ? "Failed to load subscription info" : null;

  const { mutateAsync: doUpdateTier, isPending } = useUpdateSubscriptionTier();

  async function changeTier(tier: string): Promise<string | null> {
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
        return null;
      }
      await refetch();
      return null;
    } catch (e: unknown) {
      const msg =
        e instanceof Error ? e.message : "Failed to change subscription tier";
      return msg;
    }
  }

  return {
    subscription: subscription ?? null,
    isLoading,
    error,
    isPending,
    changeTier,
  };
}
