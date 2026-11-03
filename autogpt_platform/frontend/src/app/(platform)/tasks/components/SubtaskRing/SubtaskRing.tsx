import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  done: number;
  total: number;
  size?: number;
}

const STROKE = 2;

export function SubtaskRing({ done, total, size = 16 }: Props) {
  if (total <= 0) return null;
  const radius = (size - STROKE) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(1, Math.max(0, done / total));
  const isComplete = done >= total;

  return (
    <span
      className="inline-flex items-center gap-1"
      role="img"
      aria-label={`${done} of ${total} sub-tasks done`}
      title={`${done} of ${total} sub-tasks done`}
    >
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="-rotate-90"
        aria-hidden="true"
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={STROKE}
          stroke="currentColor"
          className="text-surface-active"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={STROKE}
          stroke="currentColor"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - progress)}
          className={cn(isComplete ? "text-status-ok" : "text-info")}
        />
      </svg>
      <Text as="span" variant="small" tone="quiet" tabular>
        {done}/{total}
      </Text>
    </span>
  );
}
