"use client";

import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";

import { formatRelativeReset } from "../../../helpers";

export interface UsageWindowView {
  label: string;
  percent: number;
  prefix: string;
  value: string;
}

export function useAutopilotUsageCard() {
  const { data, isLoading } = useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsagePublic | undefined,
      refetchInterval: 30_000,
      staleTime: 10_000,
    },
  });

  const today: UsageWindowView | null = data?.daily
    ? {
        label: "Today",
        percent: Math.round(data.daily.percent_used),
        ...formatRelativeReset(data.daily.resets_at),
      }
    : null;

  const week: UsageWindowView | null = data?.weekly
    ? {
        label: "This Week",
        percent: Math.round(data.weekly.percent_used),
        ...formatRelativeReset(data.weekly.resets_at),
      }
    : null;

  return {
    today,
    week,
    isLoading,
    hasUsage: Boolean(today || week),
  };
}
