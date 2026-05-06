"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { WarningIcon } from "@phosphor-icons/react";
import {
  formatTierLabel,
  isUsageExhausted,
  TIER_BADGE_CLASS_NAME,
} from "../../usageHelpers";
import { StorageBar } from "../StorageBar";
import { UsageBar } from "../UsageBar";
import { useUsageLimitReachedCard } from "./useUsageLimitReachedCard";

export function UsageLimitReachedCard() {
  const { usage, isSuccess, isBillingEnabled } = useUsageLimitReachedCard();

  if (!isSuccess || !usage) return null;
  if (!isUsageExhausted(usage)) return null;

  const tierLabel = formatTierLabel(usage.tier);

  return (
    <div
      role="alert"
      className="mx-auto flex w-full max-w-[30rem] flex-col gap-4 rounded-2xl border border-orange-100 bg-white/70 p-4 shadow-[0_8px_32px_rgba(0,0,0,0.04)] backdrop-blur-md"
    >
      <div className="flex items-center gap-2">
        <WarningIcon className="size-5 text-orange-500" weight="fill" />
        <Text variant="body-medium" className="text-neutral-900">
          Usage limit reached
        </Text>
        {tierLabel && (
          <Badge variant="info" size="small" className={TIER_BADGE_CLASS_NAME}>
            {tierLabel} plan
          </Badge>
        )}
      </div>
      {usage.daily && (
        <UsageBar
          label="Today"
          percentUsed={usage.daily.percent_used}
          resetsAt={usage.daily.resets_at}
        />
      )}
      {usage.weekly && (
        <UsageBar
          label="This week"
          percentUsed={usage.weekly.percent_used}
          resetsAt={usage.weekly.resets_at}
        />
      )}
      <StorageBar />
      {isBillingEnabled && (
        <Button
          as="NextLink"
          href="/settings/billing"
          variant="primary"
          size="small"
          className="mt-1 w-full"
        >
          Go to billing
        </Button>
      )}
    </div>
  );
}
