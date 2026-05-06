import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

export function useUsageLimitReachedCard() {
  const { data: usage, isSuccess } = useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsagePublic,
      refetchInterval: 30000,
      staleTime: 10000,
    },
  });
  const isBillingEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);

  return { usage, isSuccess, isBillingEnabled };
}
