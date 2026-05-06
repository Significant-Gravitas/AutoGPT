import { Text } from "@/components/atoms/Text/Text";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Button } from "@/components/ui/button";
import { ChartBarIcon } from "@phosphor-icons/react";
import { StorageBar } from "../StorageBar";
import { UsageBar } from "../UsageBar";
import { useUsagePopover } from "./useUsagePopover";

export function UsagePopover() {
  const { usage, isSuccess } = useUsagePopover();

  if (!isSuccess || !usage) return null;
  if (!usage.daily && !usage.weekly) return null;

  const tierLabel = usage.tier
    ? usage.tier.charAt(0) + usage.tier.slice(1).toLowerCase()
    : null;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Usage limits">
          <ChartBarIcon className="!size-5" weight="light" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-64 p-3">
        <div className="flex flex-col gap-3">
          <div className="flex items-baseline justify-between">
            <Text
              as="span"
              variant="small-medium"
              className="font-semibold text-neutral-800"
            >
              Usage limits
            </Text>
            {tierLabel && (
              <Text as="span" variant="small" className="text-neutral-500">
                {tierLabel} plan
              </Text>
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
        </div>
      </PopoverContent>
    </Popover>
  );
}
