import type { CoPilotUsageStatus } from "@/app/api/__generated__/models/coPilotUsageStatus";
import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";

export function useUsageLimits() {
  return useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsageStatus,
      refetchInterval: 30000,
      staleTime: 10000,
    },
  });
}
