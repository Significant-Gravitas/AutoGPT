import type { CoPilotUsageStatus } from "@/app/api/__generated__/models/coPilotUsageStatus";
import Link from "next/link";

export function formatResetTime(
  resetsAt: Date | string,
  now: Date = new Date(),
): string {
  const resetDate =
    typeof resetsAt === "string" ? new Date(resetsAt) : resetsAt;
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
        <span className="text-xs font-medium text-neutral-700">{label}</span>
        <span className="text-[11px] tabular-nums text-neutral-500">
          {percentLabel}
        </span>
      </div>
      <div className="text-[10px] text-neutral-400">
        Resets {formatResetTime(resetsAt)}
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ease-out ${
            isHigh ? "bg-orange-500" : "bg-blue-500"
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
      <div className="text-xs text-neutral-500">No usage limits configured</div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="text-xs font-semibold text-neutral-800">Usage limits</div>
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
        <Link
          href="/profile/credits"
          className="text-[11px] text-blue-600 hover:underline"
        >
          Learn more about usage limits
        </Link>
      )}
    </div>
  );
}
