import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import { Button } from "@/components/atoms/Button/Button";
import Link from "next/link";
import { formatCents, formatResetTime } from "../usageHelpers";
import { useResetRateLimit } from "../../hooks/useResetRateLimit";

export { formatResetTime };

function UsageBar({
  label,
  percentUsed,
  resetsAt,
}: {
  label: string;
  percentUsed: number;
  resetsAt: Date | string;
}) {
  const percent = Math.min(100, Math.max(0, Math.round(percentUsed)));
  const isHigh = percent >= 80;
  const percentLabel =
    percentUsed > 0 && percent === 0 ? "<1% used" : `${percent}% used`;

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
          role="progressbar"
          aria-label={`${label} usage`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          className={`h-full rounded-full transition-[width] duration-300 ease-out ${
            isHigh ? "bg-orange-500" : "bg-blue-500"
          }`}
          style={{ width: `${Math.max(percent > 0 ? 1 : 0, percent)}%` }}
        />
      </div>
    </div>
  );
}

function ResetButton({
  cost,
  onCreditChange,
}: {
  cost: number;
  onCreditChange?: () => void;
}) {
  const { resetUsage, isPending } = useResetRateLimit({ onCreditChange });

  return (
    <Button
      variant="primary"
      size="small"
      onClick={() => resetUsage()}
      loading={isPending}
      className="mt-1 w-full text-[11px]"
    >
      {isPending
        ? "Resetting..."
        : `Reset daily limit for ${formatCents(cost)}`}
    </Button>
  );
}

export function UsagePanelContent({
  usage,
  showBillingLink = true,
  hasInsufficientCredits = false,
  isBillingEnabled = false,
  onCreditChange,
}: {
  usage: CoPilotUsagePublic;
  showBillingLink?: boolean;
  hasInsufficientCredits?: boolean;
  isBillingEnabled?: boolean;
  onCreditChange?: () => void;
}) {
  const daily = usage.daily;
  const weekly = usage.weekly;
  const isDailyExhausted = !!daily && daily.percent_used >= 100;
  const isWeeklyExhausted = !!weekly && weekly.percent_used >= 100;
  const resetCost = usage.reset_cost ?? 0;

  if (!daily && !weekly) {
    return (
      <div className="text-xs text-neutral-500">No usage limits configured</div>
    );
  }

  const tierLabel = usage.tier
    ? usage.tier.charAt(0) + usage.tier.slice(1).toLowerCase()
    : null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-semibold text-neutral-800">
          Usage limits
        </span>
        {tierLabel && (
          <span className="text-[11px] text-neutral-500">{tierLabel} plan</span>
        )}
      </div>
      {daily && (
        <UsageBar
          label="Today"
          percentUsed={daily.percent_used}
          resetsAt={daily.resets_at}
        />
      )}
      {weekly && (
        <UsageBar
          label="This week"
          percentUsed={weekly.percent_used}
          resetsAt={weekly.resets_at}
        />
      )}
      {isDailyExhausted &&
        !isWeeklyExhausted &&
        resetCost > 0 &&
        !hasInsufficientCredits && (
          <ResetButton cost={resetCost} onCreditChange={onCreditChange} />
        )}
      {isDailyExhausted &&
        !isWeeklyExhausted &&
        hasInsufficientCredits &&
        isBillingEnabled && (
          <Link
            href="/profile/credits"
            className="mt-1 inline-flex w-full items-center justify-center rounded-md bg-primary px-3 py-1.5 text-[11px] font-medium text-primary-foreground hover:bg-primary/90"
          >
            Add credits to reset
          </Link>
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
