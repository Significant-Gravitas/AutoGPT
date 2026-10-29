"use client";

import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";

export function useUsageIndicator() {
  const { data, isLoading } = useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsagePublic | undefined,
      refetchInterval: 30_000,
      staleTime: 10_000,
    },
  });

  const daily = data?.daily;
  const percent = daily
    ? Math.min(100, Math.max(0, Math.round(daily.percent_used)))
    : null;

  return { percent, isLoading };
}
