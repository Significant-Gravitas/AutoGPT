import { useReadEveryRender } from "@/hooks/useReadEveryRender";
import { Text } from "@/components/atoms/Text/Text";
import { formatResetTime } from "../usageHelpers";

interface Props {
  label: string;
  percentUsed: number;
  resetsAt: Date | string;
}

export function UsageBar({ label, percentUsed, resetsAt }: Props) {
  const percent = Math.min(100, Math.max(0, Math.round(percentUsed)));
  const isHigh = percent >= 80;
  const percentLabel =
    percentUsed > 0 && percent === 0 ? "<1% used" : `${percent}% used`;
  // Relative to the clock, so read on every render rather than cached.
  const resetLabel = useReadEveryRender(() => formatResetTime(resetsAt));

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <Text tone="body" variant="body-medium">
          {label}
        </Text>
        <Text tone="muted" variant="body" className="tabular-nums">
          {percentLabel}
        </Text>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-surface-active">
        <div
          role="progressbar"
          aria-label={`${label} usage`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          className={`h-full rounded-full transition-[width] duration-300 ease-out ${
            // oxlint-disable-next-line design-system/no-raw-colors -- orange "almost full" bar; warning is amber, a hue change deferred
            isHigh ? "bg-orange-500" : "bg-info"
          }`}
          style={{ width: `${Math.max(percent > 0 ? 1 : 0, percent)}%` }}
        />
      </div>
      <Text tone="muted" variant="small">
        Resets {resetLabel}
      </Text>
    </div>
  );
}
