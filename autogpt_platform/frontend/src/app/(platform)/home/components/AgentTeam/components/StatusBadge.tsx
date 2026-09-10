import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import { cn } from "@/lib/utils";

const STATUS_CONFIG = {
  working: { label: "Working", dot: "bg-primary", pulse: true },
  ready: { label: "Ready", dot: "bg-emerald-500", pulse: false },
  needs_setup: { label: "Finish setup", dot: "bg-amber-500", pulse: false },
  paused: { label: "Paused", dot: "bg-zinc-400", pulse: false },
  failed: { label: "Failed", dot: "bg-red-500", pulse: false },
};

const UNKNOWN_STATUS_CONFIG = {
  label: "Unknown",
  dot: "bg-zinc-300",
  pulse: false,
};

interface Props {
  status: HomeAgentStatus["status"];
}

/** A dot and a word: the status reads at a glance without a filled pill
 *  competing with the name beside it. */
export function StatusBadge({ status }: Props) {
  const config = STATUS_CONFIG[status] ?? UNKNOWN_STATUS_CONFIG;

  return (
    <span className="inline-flex shrink-0 items-center gap-1.5 text-[11px] font-medium text-zinc-500">
      <span
        className={cn(
          "size-1.5 rounded-full",
          config.dot,
          config.pulse && "animate-pulse",
        )}
        aria-hidden="true"
      />
      {config.label}
    </span>
  );
}
