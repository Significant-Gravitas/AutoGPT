import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatResetTime } from "../usageHelpers";

type Size = "sm" | "md";

interface Props {
  label: string;
  percentUsed: number;
  resetsAt: Date | string;
  size?: Size;
}

export function UsageBar({ label, percentUsed, resetsAt, size = "sm" }: Props) {
  const percent = Math.min(100, Math.max(0, Math.round(percentUsed)));
  const percentLabel =
    percentUsed > 0 && percent === 0 ? "<1% used" : `${percent}% used`;
  const labelVariant = size === "md" ? "body-medium" : "small-medium";

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <Text as="span" variant={labelVariant} className="text-neutral-700">
          {label}
        </Text>
        <Text
          as="span"
          variant="small"
          className="tabular-nums text-neutral-500"
        >
          {percentLabel}
        </Text>
      </div>
      <Text as="span" variant="small" className="text-neutral-400">
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
