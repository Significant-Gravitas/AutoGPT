import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

const STATUS_CONFIG = {
  working: { label: "Working", dot: "bg-primary", pulse: true },
  ready: { label: "Ready", dot: "bg-success", pulse: false },
  needs_setup: { label: "Finish setup", dot: "bg-warning", pulse: false },
  paused: { label: "Paused", dot: "bg-foreground-subtle", pulse: false },
  failed: { label: "Failed", dot: "bg-error", pulse: false },
};

const UNKNOWN_STATUS_CONFIG = {
  label: "Unknown",
  dot: "bg-border-strong",
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
    <Text
      variant="small-medium"
      as="span"
      tone="muted"
      className="inline-flex shrink-0 items-center gap-1.5"
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          config.dot,
          config.pulse && "animate-pulse",
        )}
        aria-hidden="true"
      />
      {config.label}
    </Text>
  );
}
