"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import Link from "next/link";
import type { ReactNode } from "react";
import { formatTierLabel, TIER_BADGE_CLASS_NAME } from "../../usageHelpers";
import { UsageMeter } from "@/components/organisms/UsageExperience/UsageMeter";
import { formatUsageDate } from "@/services/usageExperience/helpers";
import { StorageBar } from "../StorageBar";
import { UsageBar } from "../UsageBar";
import { useUsagePopover } from "./useUsagePopover";
import { GaugeIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  trigger?: ReactNode;
  align?: "start" | "center" | "end";
}

export function UsagePopover({ trigger, align = "start" }: Props) {
  const { usage, experience, isLoading, isError, retry, isBillingEnabled } =
    useUsagePopover();

  if (isLoading) return null;
  if (!isError && !usage?.daily && !usage?.weekly && !experience.isActiveTrial)
    return null;

  const tierLabel = formatTierLabel(experience.tier);

  return (
    <Popover>
      <PopoverTrigger asChild>
        {trigger ?? (
          <Button variant="ghost" size="icon" aria-label="Usage limits">
            <Icon icon={GaugeIcon} className="!size-5" />
          </Button>
        )}
      </PopoverTrigger>
      {/* z-[80]: must layer above the Otto mobile drawer
          (overlay z-[60], content z-[70] in MobileDrawer.tsx) so the
          popover doesn't render under the drawer's blur. */}
      <PopoverContent align={align} className="z-[80] w-72 p-4">
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
                {tierLabel}
              </Badge>
            )}
          </div>
          {isError && (
            <div>
              <Text variant="small">We couldn’t check your usage.</Text>
              <Button size="small" variant="ghost" onClick={() => void retry()}>
                Try again
              </Button>
            </div>
          )}
          {!isError &&
            experience.isActiveTrial &&
            experience.trialPercent !== null && (
              <UsageMeter
                label="Trial allowance"
                percent={experience.trialPercent}
                detail={
                  experience.trialEndsAt
                    ? `Trial ends ${formatUsageDate(experience.trialEndsAt)}`
                    : "Your allowance for this trial"
                }
              />
            )}
          {!isError && !experience.isLifetimeTrial && usage?.daily && (
            <UsageBar
              label="Today"
              percentUsed={usage.daily.percent_used}
              resetsAt={usage.daily.resets_at}
            />
          )}
          {!isError && !experience.isLifetimeTrial && usage?.weekly && (
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
