import type { CoPilotUsageStatus } from "@/app/api/__generated__/models/coPilotUsageStatus";
import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";

export function useUsageLimits(sessionID: string | null) {
  return useGetV2GetCopilotUsage(
    sessionID ? { session_id: sessionID } : undefined,
    {
      query: {
        select: (res) => res.data as CoPilotUsageStatus,
        enabled: !!sessionID,
        refetchInterval: 30000,
        staleTime: 10000,
      },
    },
  );
}
