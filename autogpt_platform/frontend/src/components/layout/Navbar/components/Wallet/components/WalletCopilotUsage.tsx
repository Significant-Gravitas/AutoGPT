"use client";

import { UsageBar } from "@/app/(platform)/copilot/components/UsageLimits/UsageBar";
import { useCopilotUsage } from "@/app/(platform)/copilot/components/UsageLimits/useCopilotUsage";
import {
  formatTierLabel,
  TIER_BADGE_CLASS_NAME,
} from "@/app/(platform)/copilot/components/usageHelpers";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Progress } from "@/components/atoms/Progress/Progress";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { formatTrialEnd } from "@/components/organisms/TrialCard/helpers";
import { cn } from "@/lib/utils";
import { useTrialStatus } from "@/services/trials/useTrialStatus";

export function WalletCopilotUsage() {
  const { data: usage, isError, refetch } = useCopilotUsage();
  const tierLabel = formatTierLabel(usage?.tier);

  return (
    <section aria-label="Chat usage" className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between gap-3">
          <Text variant="large-medium" as="h2">
            Chat usage
          </Text>
          {tierLabel && (
            <Badge
              variant="info"
              size="small"
              className={TIER_BADGE_CLASS_NAME}
            >
              {tierLabel === "Trial" ? "Free trial" : `${tierLabel} plan`}
            </Badge>
          )}
        </div>
        <Text variant="small" tone="secondary">
          Your allowance for conversations with Otto and experts.
        </Text>
      </div>
      {usage ? (
        <UsageWindows usage={usage} />
      ) : isError ? (
        <UsageUnavailable onRetry={() => void refetch()} />
      ) : (
        <UsageLoading />
      )}
    </section>
  );
}

function UsageWindows({ usage }: { usage: CoPilotUsagePublic }) {
  if (usage.tier === "TRIAL") return <TrialUsage />;
  if (!usage.daily && !usage.weekly) {
    return (
      <Text variant="body" tone="secondary">
        No usage limits
      </Text>
    );
  }

  return (
    <div className="flex flex-col gap-4">
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
    </div>
  );
}

function TrialUsage() {
  const {
    data: trial,
    isPending,
    refetch,
  } = useTrialStatus({
    refetchInterval: 30_000,
  });
  if (isPending) return <UsageLoading />;
  if (trial?.allowance_used_percent == null) {
    return <UsageUnavailable onRetry={() => void refetch()} />;
  }

  const percent = Math.min(100, Math.max(0, trial.allowance_used_percent));

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-3">
        <Text variant="body-medium" className="text-neutral-700">
          Trial allowance
        </Text>
        <Text variant="body" className="tabular-nums text-neutral-500">
          {formatTrialPercent(percent)} used
        </Text>
      </div>
      <Progress
        value={percent}
        role="progressbar"
        aria-label="Trial allowance usage"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        className={cn(
          "bg-neutral-200 [&>div]:rounded-full",
          percent >= 80 ? "[&>div]:bg-orange-500" : "[&>div]:bg-blue-500",
        )}
      />
      <Text variant="small" className="text-neutral-500">
        {trial.active ? "Ends" : "Ended"} {formatTrialEnd(trial.ends_at)}
      </Text>
    </div>
  );
}

function formatTrialPercent(percent: number) {
  if (percent > 0 && percent < 0.5) return "<1%";
  if (percent >= 99.5 && percent < 100) {
    return `${Math.floor(percent * 10) / 10}%`;
  }
  return `${Math.round(percent)}%`;
}

function UsageLoading() {
  return (
    <div
      role="status"
      aria-label="Loading chat usage"
      className="flex flex-col gap-4"
    >
      {[0, 1].map((index) => (
        <div key={index} aria-hidden="true" className="flex flex-col gap-2">
          <div className="flex justify-between">
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-4 w-14" />
          </div>
          <Skeleton className="h-2 w-full rounded-full" />
          <Skeleton className="h-3 w-24" />
        </div>
      ))}
    </div>
  );
}

function UsageUnavailable({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <Text variant="small" tone="secondary">
        Usage is unavailable right now.
      </Text>
      <Button
        variant="ghost"
        size="small"
        onClick={onRetry}
        className="h-auto px-2 py-1 text-xs"
      >
        Retry
      </Button>
    </div>
  );
}
