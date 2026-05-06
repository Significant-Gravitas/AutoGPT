import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";

export function useIsUsageLimitReached() {
  const { data: usage } = useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsagePublic,
      refetchInterval: 30000,
      staleTime: 10000,
    },
  });
  const daily = usage?.daily?.percent_used ?? 0;
  const weekly = usage?.weekly?.percent_used ?? 0;
  return daily >= 100 || weekly >= 100;
}
