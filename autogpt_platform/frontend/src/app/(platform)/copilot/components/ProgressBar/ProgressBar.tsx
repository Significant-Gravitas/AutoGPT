import { cn } from "@/lib/utils";

interface Props {
  value: number;
  label?: string;
  className?: string;
}

export function ProgressBar({ value, label, className }: Props) {
  const clamped = Math.min(100, Math.max(0, value));

  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div className="flex items-center justify-between text-xs text-neutral-500">
        <span>{label ?? "Working on it..."}</span>
        <span>{Math.round(clamped)}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200">
        <div
          className="h-full rounded-full bg-neutral-900 transition-[width] duration-300 ease-out"
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}
