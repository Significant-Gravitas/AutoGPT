import type { CoPilotUsageStatus } from "@/app/api/__generated__/models/coPilotUsageStatus";
import { useUsageLimits } from "./useUsageLimits";

function formatResetTime(resetsAt: Date): string {
  const now = new Date();
  const diffMs = resetsAt.getTime() - now.getTime();
  if (diffMs <= 0) return "now";

  const hours = Math.floor(diffMs / (1000 * 60 * 60));
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
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
  resetsAt: Date;
}) {
  if (limit <= 0) return null;

  const percent = Math.min(100, Math.round((used / limit) * 100));
  const isHigh = percent >= 80;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
          {label}
        </span>
        <span className="text-[11px] tabular-nums text-neutral-500 dark:text-neutral-400">
          {percent}% used
        </span>
      </div>
      <div className="text-[10px] text-neutral-400 dark:text-neutral-500">
        Resets in {formatResetTime(resetsAt)}
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-700">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ease-out ${
            isHigh ? "bg-orange-500" : "bg-blue-500"
          }`}
          style={{ width: `${percent}%` }}
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
          Manage billing &amp; credits
        </a>
      )}
    </div>
  );
}

export function UsageLimits() {
  const { data: usage, isLoading } = useUsageLimits();

  if (isLoading || !usage) return null;
  if (usage.daily.limit <= 0 && usage.weekly.limit <= 0) return null;

  return <UsagePanelContent usage={usage} />;
}
