"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { ChartBarIcon } from "@phosphor-icons/react";
import Link from "next/link";
import { formatTierLabel, TIER_BADGE_CLASS_NAME } from "../../usageHelpers";
import { StorageBar } from "../StorageBar";
import { UsageBar } from "../UsageBar";
import { useUsagePopover } from "./useUsagePopover";

export function UsagePopover() {
  const { usage, isSuccess, isBillingEnabled } = useUsagePopover();

  if (!isSuccess || !usage) return null;
  if (!usage.daily && !usage.weekly) return null;

  const tierLabel = formatTierLabel(usage.tier);

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Usage limits">
          <ChartBarIcon className="!size-5" weight="light" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-72 p-4">
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <Text variant="body-medium" className="text-neutral-800">
              Usage limits
            </Text>
            {tierLabel && (
              <Badge
                variant="info"
                size="small"
                className={TIER_BADGE_CLASS_NAME}
              >
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
            <Link
              href="/settings/billing"
              className="text-sm text-blue-600 hover:underline"
            >
              Manage billing
            </Link>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
