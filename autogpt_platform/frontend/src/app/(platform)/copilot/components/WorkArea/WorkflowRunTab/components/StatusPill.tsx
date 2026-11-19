import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { statusView, type RunTone } from "../runs";

const TONE_CLASS: Record<RunTone, string> = {
  live: "bg-info/10 text-info",
  ok: "bg-status-ok/10 text-status-ok",
  failed: "bg-error/10 text-error",
  idle: "bg-surface-wash text-foreground-quiet",
  waiting: "bg-status-waiting/15 text-warning-foreground",
};

interface Props {
  status: string | undefined;
}

/** The run's state as a small pill; a live run's dot breathes. */
export function StatusPill({ status }: Props) {
  const view = statusView(status);
  return (
    <span
      data-testid="run-status"
      className={cn(
        "inline-flex h-5 shrink-0 items-center gap-1.5 rounded-full px-2",
        TONE_CLASS[view.tone],
      )}
    >
      <span
        aria-hidden
        className={cn(
          "size-1.5 rounded-full bg-current",
          view.tone === "live" && "animate-pulse motion-reduce:animate-none",
        )}
      />
      <Text variant="small-medium" as="span" tone="inherit">
        {view.label}
      </Text>
    </span>
  );
}
