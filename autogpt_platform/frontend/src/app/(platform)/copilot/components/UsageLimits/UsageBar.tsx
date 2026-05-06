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

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <Text variant="body-medium" className="text-neutral-700">
          {label}
        </Text>
        <Text variant="body" className="tabular-nums text-neutral-500">
          {percentLabel}
        </Text>
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
      <Text variant="small" className="text-neutral-400">
        Resets {formatResetTime(resetsAt)}
      </Text>
    </div>
  );
}
