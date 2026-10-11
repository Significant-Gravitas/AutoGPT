import { cn } from "@/lib/utils";
import { UsageMeter } from "./UsageMeter";

interface Reading {
  percent_used: number;
}

interface Props {
  daily?: Reading | null;
  weekly?: Reading | null;
  /** A lifetime trial has one allowance instead of daily and weekly ones. */
  trialPercent?: number | null;
  className?: string;
}

export function UsageMeters({ daily, weekly, trialPercent, className }: Props) {
  const meters =
    trialPercent != null
      ? [{ label: "Trial allowance", percent: trialPercent }]
      : [
          ...(daily ? [{ label: "Today", percent: daily.percent_used }] : []),
          ...(weekly
            ? [{ label: "This week", percent: weekly.percent_used }]
            : []),
        ];
  if (meters.length === 0) return null;
  return (
    <div className={cn("grid gap-x-6 gap-y-3 sm:grid-cols-2", className)}>
      {meters.map((meter) => (
        <UsageMeter
          key={meter.label}
          label={meter.label}
          percent={meter.percent}
        />
      ))}
    </div>
  );
}
