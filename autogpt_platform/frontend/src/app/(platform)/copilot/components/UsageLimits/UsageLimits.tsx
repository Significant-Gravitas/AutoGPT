import type { CoPilotUsageStatus } from "@/app/api/__generated__/models/coPilotUsageStatus";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { ChartBar } from "@phosphor-icons/react";
import { useUsageLimits } from "./useUsageLimits";

interface Props {
  sessionID: string | null;
}

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

function UsagePanelContent({ usage }: { usage: CoPilotUsageStatus }) {
  const hasSessionLimit = usage.session.limit > 0;
  const hasWeeklyLimit = usage.weekly.limit > 0;

  if (!hasSessionLimit && !hasWeeklyLimit) {
    return (
      <div className="text-xs text-neutral-500 dark:text-neutral-400">
        No usage limits configured
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="text-xs font-semibold text-neutral-800 dark:text-neutral-200">
        Plan usage limits
      </div>
      {hasSessionLimit && (
        <UsageBar
          label="Current session"
          used={usage.session.used}
          limit={usage.session.limit}
          resetsAt={usage.session.resets_at}
        />
      )}
      {hasWeeklyLimit && (
        <UsageBar
          label="Weekly limits"
          used={usage.weekly.used}
          limit={usage.weekly.limit}
          resetsAt={usage.weekly.resets_at}
        />
      )}
      <a
        href="/profile/credits"
        className="text-[11px] text-blue-600 hover:underline dark:text-blue-400"
      >
        Learn more about usage limits
      </a>
    </div>
  );
}

export function UsageLimits({ sessionID }: Props) {
  const { data: usage, isLoading } = useUsageLimits(sessionID);

  // Don't show if no limits configured or still loading
  if (isLoading || !usage) return null;
  if (usage.session.limit <= 0 && usage.weekly.limit <= 0) return null;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          className="rounded p-1.5 hover:bg-neutral-100 dark:hover:bg-neutral-800"
          aria-label="Usage limits"
        >
          <ChartBar className="h-4 w-4 text-neutral-500 dark:text-neutral-400" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72 p-4">
        <UsagePanelContent usage={usage} />
      </PopoverContent>
    </Popover>
  );
}
