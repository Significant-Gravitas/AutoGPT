import { cn } from "@/lib/utils";

const ORIGIN_STYLES: Record<string, { label: string; className: string }> = {
  USER: { label: "You", className: "bg-sky-50 text-sky-700 ring-sky-200" },
  SCHEDULE: {
    label: "Schedule",
    className: "bg-violet-50 text-violet-700 ring-violet-200",
  },
  DREAM: {
    label: "Proactive",
    className: "bg-fuchsia-50 text-fuchsia-700 ring-fuchsia-200",
  },
  EXPERT: {
    label: "Expert",
    className: "bg-teal-50 text-teal-700 ring-teal-200",
  },
};

interface Props {
  createdByType: string | null | undefined;
  className?: string;
}

export function OriginBadge({ createdByType, className }: Props) {
  const style = createdByType ? ORIGIN_STYLES[createdByType] : undefined;
  if (!style) return null;

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset",
        style.className,
        className,
      )}
    >
      {style.label}
    </span>
  );
}
