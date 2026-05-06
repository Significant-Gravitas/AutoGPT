import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { formatResetTime } from "../usageHelpers";
import { useWorkspaceStorage } from "./useWorkspaceStorage";

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

export function formatBytes(bytes: number): string {
  const KB = 1024;
  const MB = KB * 1024;
  const GB = MB * 1024;
  if (bytes < KB) return `${bytes} B`;
  if (bytes < MB) {
    const kb = Math.round(bytes / KB);
    return kb >= 1024 ? `${(bytes / MB).toFixed(1)} MB` : `${kb} KB`;
  }
  if (bytes < GB) {
    const mb = Math.round(bytes / MB);
    return mb >= 1024 ? `${(bytes / GB).toFixed(1)} GB` : `${mb} MB`;
  }
  return `${(bytes / GB).toFixed(1)} GB`;
}

function StorageBar({
  usedBytes,
  limitBytes,
  fileCount,
}: {
  usedBytes: number;
  limitBytes: number;
  fileCount: number;
}) {
  if (limitBytes <= 0) return null;

  const rawPercent = (usedBytes / limitBytes) * 100;
  const percent = Math.min(100, Math.round(rawPercent));
  const isHigh = percent >= 80;
  const percentLabel =
    usedBytes > 0 && percent === 0 ? "<1% used" : `${percent}% used`;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-neutral-700">
          File storage
        </span>
        <span className="text-[11px] tabular-nums text-neutral-500">
          {percentLabel}
        </span>
      </div>
      <div className="text-[10px] text-neutral-400">
        {formatBytes(usedBytes)} of {formatBytes(limitBytes)} &middot;{" "}
        {fileCount} {fileCount === 1 ? "file" : "files"}
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ease-out ${
            isHigh ? "bg-orange-500" : "bg-blue-500"
          }`}
          style={{ width: `${Math.max(usedBytes > 0 ? 1 : 0, percent)}%` }}
        />
      </div>
    </div>
  );
}

function WorkspaceStorageSection() {
  const { data: storage } = useWorkspaceStorage();
  if (!storage || storage.limit_bytes <= 0) return null;

  return (
    <StorageBar
      usedBytes={storage.used_bytes}
      limitBytes={storage.limit_bytes}
      fileCount={storage.file_count}
    />
  );
}

export function UsagePanelContent({
  usage,
  showHeader = true,
  showBillingLink = true,
  size = "sm",
}: {
  usage: CoPilotUsagePublic;
  showHeader?: boolean;
  showBillingLink?: boolean;
  size?: Size;
}) {
  const isBillingEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const daily = usage.daily;
  const weekly = usage.weekly;
  const isDailyExhausted = !!daily && daily.percent_used >= 100;

  if (!daily && !weekly) {
    return (
      <div className="flex flex-col gap-3">
        <Text as="span" variant="small" className="text-neutral-500">
          No usage limits configured
        </Text>
        <WorkspaceStorageSection />
      </div>
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
      <WorkspaceStorageSection />
      {isDailyExhausted && isBillingEnabled && showBillingLink && (
        <Button
          as="NextLink"
          href="/settings/billing"
          variant="primary"
          size="small"
          className="mt-1 w-full"
        >
          Go to billing
        </Button>
      )}
      {showBillingLink && !(isDailyExhausted && isBillingEnabled) && (
        <Link href="/settings/billing" className="hover:underline">
          <Text as="span" variant="small" className="text-blue-600">
            Learn more about usage limits
          </Text>
        </Link>
      )}
    </div>
  );
}
