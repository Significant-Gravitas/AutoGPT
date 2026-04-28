"use client";

import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import useCredits from "@/hooks/useCredits";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { Warning } from "@phosphor-icons/react";
import Link from "next/link";
import { UsagePanelContent } from "./UsagePanelContent";

const USAGE_QUERY_CONFIG = {
  query: {
    select: (res: { data: unknown }) => res.data as CoPilotUsagePublic,
    refetchInterval: 30000,
    staleTime: 10000,
  },
} as const;

export function useIsUsageLimitReached() {
  const { data: usage } = useGetV2GetCopilotUsage(USAGE_QUERY_CONFIG);
  const daily = usage?.daily?.percent_used ?? 0;
  const weekly = usage?.weekly?.percent_used ?? 0;
  return daily >= 100 || weekly >= 100;
}

export function UsageLimitReachedCard() {
  const { data: usage, isSuccess } =
    useGetV2GetCopilotUsage(USAGE_QUERY_CONFIG);

  const isBillingEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const { credits, fetchCredits } = useCredits({ fetchInitialCredits: true });
  const resetCost = usage?.reset_cost;
  const hasInsufficientCredits =
    credits !== null && resetCost != null && credits < resetCost;

  if (!isSuccess || !usage) return null;
  const isDailyExhausted = !!usage.daily && usage.daily.percent_used >= 100;
  const isWeeklyExhausted = !!usage.weekly && usage.weekly.percent_used >= 100;
  if (!isDailyExhausted && !isWeeklyExhausted) return null;

  const tierLabel = usage.tier
    ? usage.tier.charAt(0) + usage.tier.slice(1).toLowerCase()
    : null;

  return (
    <div
      role="alert"
      className="mx-auto flex w-full max-w-[30rem] flex-col gap-3 rounded-2xl border border-orange-100 bg-white/70 p-4 shadow-[0_8px_32px_rgba(0,0,0,0.04)] backdrop-blur-md"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Warning className="size-5 text-orange-500" weight="fill" />
          <Text
            as="span"
            variant="body-medium"
            className="font-semibold text-neutral-900"
          >
            Usage limit reached
          </Text>
          {tierLabel && (
            <Badge
              variant="info"
              size="small"
              className="border border-neutral-200"
            >
              {tierLabel}
            </Badge>
          )}
        </div>
        <Link href="/profile/credits" className="shrink-0 hover:underline">
          <Text as="span" variant="small" className="text-blue-600">
            Learn more about usage limits
          </Text>
        </Link>
      </div>
      <UsagePanelContent
        usage={usage}
        showHeader={false}
        showBillingLink={false}
        hasInsufficientCredits={hasInsufficientCredits}
        isBillingEnabled={isBillingEnabled}
        onCreditChange={fetchCredits}
        size="md"
      />
    </div>
  );
}
