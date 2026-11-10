import { useUpdateSubscriptionTier } from "@/app/api/__generated__/endpoints/credits/credits";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useCheckoutRedirect } from "@/hooks/useCheckoutRedirect";
import { getCheckoutReturnURLs } from "../SubscriptionTab/helpers";
import type { AbandonedPlan } from "./helpers";

// Opens the checkout the person left again, for someone whose card failed.
export function useCheckoutRetry(plan: AbandonedPlan | null) {
  const { mutateAsync: updateTier, isPending } = useUpdateSubscriptionTier();
  const checkoutRedirect = useCheckoutRedirect();

  async function retry() {
    if (!plan || isPending || checkoutRedirect.isRedirecting) return;
    const { successURL, cancelURL } = getCheckoutReturnURLs(plan);
    try {
      const response = await updateTier({
        data: {
          tier: plan.tier,
          billing_cycle: plan.cycle,
          success_url: successURL,
          cancel_url: cancelURL,
          surface: "billing",
        },
      });
      const url = response.status === 200 ? response.data.url : undefined;
      if (!url) throw new Error("No checkout to open");
      window.location.href = url;
      checkoutRedirect.markRedirecting();
    } catch {
      toast({
        title: "Couldn't open checkout",
        description: "Pick your plan again below to retry.",
        variant: "destructive",
      });
    }
  }

  return {
    retry: plan ? retry : undefined,
    isRetrying: isPending || checkoutRedirect.isRedirectingHere,
  };
}
