import { Text } from "@/components/atoms/Text/Text";
import { formatUsagePercent } from "@/services/usageExperience/helpers";

interface Props {
  label: string;
  percent: number;
  detail?: string | null;
}

export function UsageMeter({ label, percent, detail }: Props) {
  const value = Math.min(100, Math.max(0, percent));
  return (
    <div className="space-y-2">
      <div className="flex justify-between gap-2">
        <Text variant="small-medium" className="!text-zinc-700">
          {label}
        </Text>
        <Text variant="small" className="tabular-nums !text-zinc-500">
          {formatUsagePercent(value)} used
        </Text>
      </div>
      <div
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={value}
        className="h-1.5 overflow-hidden rounded-full bg-zinc-100"
      >
        <div
          className="h-full rounded-full bg-purple-500 transition-[width]"
          style={{ width: `${value}%` }}
        />
      </div>
      {detail && (
        <Text variant="small" className="!text-xs !text-zinc-500">
          {detail}
        </Text>
      )}
    </div>
  );
}
