import type { CSSProperties } from "react";
import { Text } from "@/components/atoms/Text/Text";
import { formatUsagePercent } from "@/services/usageExperience/helpers";
import { cn } from "@/lib/utils";

interface Props {
  label: string;
  percent: number;
  detail?: string | null;
}

export function UsageMeter({ label, percent, detail }: Props) {
  const value = Math.min(100, Math.max(0, percent));
  return (
    <div data-slot="usage-meter" className="space-y-2">
      <div
        data-slot="usage-meter-header"
        className="flex justify-between gap-2"
      >
        <Text variant="small-medium" className="!text-foreground-body">
          {label}
        </Text>
        <Text variant="small" tone="muted" className="tabular-nums">
          {formatUsagePercent(value)} used
        </Text>
      </div>
      <div
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={value}
        data-slot="usage-meter-track"
        className="h-1.5 overflow-hidden rounded-full bg-surface-hover"
      >
        <div
          data-slot="usage-meter-fill"
          className={cn(
            "w-(--usage-meter-width) h-full rounded-full transition-[width]",
            // TODO(tokens): no role for orange-500, the full-meter fill.
            value >= 100 ? "bg-orange-500" : "bg-brand",
          )}
          style={{ "--usage-meter-width": `${value}%` } as CSSProperties}
        />
      </div>
      {detail && (
        <Text variant="small" tone="muted" className="!text-xs">
          {detail}
        </Text>
      )}
    </div>
  );
}
