import type { CoPilotUsageStatus } from "@/app/api/__generated__/models/coPilotUsageStatus";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Button } from "@/components/ui/button";
import { ChartBar } from "@phosphor-icons/react";
import { useUsageLimits } from "./useUsageLimits";

function formatResetTime(resetsAt: Date | string): string {
  const resetDate =
    typeof resetsAt === "string" ? new Date(resetsAt) : resetsAt;
  const now = new Date();
  const diffMs = resetDate.getTime() - now.getTime();
  if (diffMs <= 0) return "now";

  const hours = Math.floor(diffMs / (1000 * 60 * 60));

  // Under 24h: show relative time ("in 4h 23m")
  if (hours < 24) {
    const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
    if (hours > 0) return `in ${hours}h ${minutes}m`;
    return `in ${minutes}m`;
  }

  // Over 24h: show day and time in local timezone ("Mon 12:00 AM PST")
  return resetDate.toLocaleString(undefined, {
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

function UsageBar({
  label,
  used,
  limit,
  resetsAt,
}: {
  label: string;
  used: number;
  limit: number;
  resetsAt: Date | string;
}) {
  if (limit <= 0) return null;

  const rawPercent = (used / limit) * 100;
  const percent = Math.min(100, Math.round(rawPercent));
  const isHigh = percent >= 80;
  const percentLabel =
    used > 0 && percent === 0 ? "<1% used" : `${percent}% used`;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
          {label}
        </span>
        <span className="text-[11px] tabular-nums text-neutral-500 dark:text-neutral-400">
          {percentLabel}
        </span>
      </div>
      <div className="text-[10px] text-neutral-400 dark:text-neutral-500">
        Resets {formatResetTime(resetsAt)}
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-700">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ease-out ${
            isHigh
              ? "bg-orange-500 dark:bg-orange-400"
              : "bg-blue-500 dark:bg-blue-400"
          }`}
          style={{ width: `${Math.max(used > 0 ? 1 : 0, percent)}%` }}
        />
      </div>
    </div>
  );
}

export function UsagePanelContent({
  usage,
  showBillingLink = true,
}: {
  usage: CoPilotUsageStatus;
  showBillingLink?: boolean;
}) {
  const hasDailyLimit = usage.daily.limit > 0;
  const hasWeeklyLimit = usage.weekly.limit > 0;

  if (!hasDailyLimit && !hasWeeklyLimit) {
    return (
      <div className="text-xs text-neutral-500 dark:text-neutral-400">
        No usage limits configured
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="text-xs font-semibold text-neutral-800 dark:text-neutral-200">
        Usage limits
      </div>
      {hasDailyLimit && (
        <UsageBar
          label="Today"
          used={usage.daily.used}
          limit={usage.daily.limit}
          resetsAt={usage.daily.resets_at}
        />
      )}
      {hasWeeklyLimit && (
        <UsageBar
          label="This week"
          used={usage.weekly.used}
          limit={usage.weekly.limit}
          resetsAt={usage.weekly.resets_at}
        />
      )}
      {showBillingLink && (
        <a
          href="/profile/credits"
          className="text-[11px] text-blue-600 hover:underline dark:text-blue-400"
        >
          Learn more about usage limits
        </a>
      )}
    </div>
  );
}

export function UsageLimits() {
  const { data: usage, isLoading } = useUsageLimits();

  if (isLoading || !usage) return null;
  if (usage.daily.limit <= 0 && usage.weekly.limit <= 0) return null;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Usage limits">
          <ChartBar className="!size-5" weight="light" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-64 p-3">
        <UsagePanelContent usage={usage} />
      </PopoverContent>
    </Popover>
  );
}
