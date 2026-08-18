import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";

const USAGE_QUERY_CONFIG = {
  query: {
    select: (res: { data: unknown }) => res.data as CoPilotUsagePublic,
    refetchInterval: 30000,
    staleTime: 10000,
  },
} as const;

export function useCopilotUsage() {
  return useGetV2GetCopilotUsage(USAGE_QUERY_CONFIG);
}
