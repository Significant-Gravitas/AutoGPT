import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { formatCents, formatResetTime } from "../usageHelpers";
import { useResetRateLimit } from "../../hooks/useResetRateLimit";

export { formatResetTime };

type Size = "sm" | "md";

const labelVariant = (size: Size) =>
  size === "md" ? "body-medium" : "small-medium";
const metaVariant = "small" as const;

function UsageBar({
  label,
  percentUsed,
  resetsAt,
  size = "sm",
}: {
  label: string;
  percentUsed: number;
  resetsAt: Date | string;
  size?: Size;
}) {
  const percent = Math.min(100, Math.max(0, Math.round(percentUsed)));
  const percentLabel =
    percentUsed > 0 && percent === 0 ? "<1% used" : `${percent}% used`;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <Text
          as="span"
          variant={labelVariant(size)}
          className="text-neutral-700"
        >
          {label}
        </Text>
        <Text
          as="span"
          variant={metaVariant}
          className="tabular-nums text-neutral-500"
        >
          {percentLabel}
        </Text>
      </div>
      <Text as="span" variant={metaVariant} className="text-neutral-400">
        Resets {formatResetTime(resetsAt)}
      </Text>
      <div
        className={cn(
          "w-full overflow-hidden rounded-full bg-neutral-200",
          size === "md" ? "h-2.5" : "h-2",
        )}
      >
        <div
          role="progressbar"
          aria-label={`${label} usage`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          className="h-full rounded-full bg-blue-500 transition-[width] duration-300 ease-out"
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
      className="mt-1 w-full"
    >
      {isPending
        ? "Resetting..."
        : `Reset daily limit for ${formatCents(cost)}`}
    </Button>
  );
}

export function UsagePanelContent({
  usage,
  showHeader = true,
  showBillingLink = true,
  hasInsufficientCredits = false,
  isBillingEnabled = false,
  onCreditChange,
  size = "sm",
}: {
  usage: CoPilotUsagePublic;
  showHeader?: boolean;
  showBillingLink?: boolean;
  hasInsufficientCredits?: boolean;
  isBillingEnabled?: boolean;
  onCreditChange?: () => void;
  size?: Size;
}) {
  const daily = usage.daily;
  const weekly = usage.weekly;
  const isDailyExhausted = !!daily && daily.percent_used >= 100;
  const isWeeklyExhausted = !!weekly && weekly.percent_used >= 100;
  const resetCost = usage.reset_cost ?? 0;

  if (!daily && !weekly) {
    return (
      <Text as="span" variant="small" className="text-neutral-500">
        No usage limits configured
      </Text>
    );
  }

  const tierLabel = usage.tier
    ? usage.tier.charAt(0) + usage.tier.slice(1).toLowerCase()
    : null;

  return (
    <div className="flex flex-col gap-3">
      {showHeader && (
        <div className="flex items-baseline justify-between">
          <Text
            as="span"
            variant={size === "md" ? "body-medium" : "small-medium"}
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
      )}
      {daily && (
        <UsageBar
          label="Today"
          percentUsed={daily.percent_used}
          resetsAt={daily.resets_at}
          size={size}
        />
      )}
      {weekly && (
        <UsageBar
          label="This week"
          percentUsed={weekly.percent_used}
          resetsAt={weekly.resets_at}
          size={size}
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
          <Button
            as="NextLink"
            href="/profile/credits"
            variant="primary"
            size="small"
            className="mt-1 w-full"
          >
            Add credits to reset
          </Button>
        )}
      {showBillingLink && (
        <Link href="/profile/credits" className="hover:underline">
          <Text as="span" variant="small" className="text-blue-600">
            Learn more about usage limits
          </Text>
        </Link>
      )}
    </div>
  );
}
