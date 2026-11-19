import { Clock01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  show?: boolean;
  onRetry?: () => void;
  retrying?: boolean;
  /** Button under the text whatever the viewport, for a narrow column. */
  stacked?: boolean;
  className?: string;
}

// The scheduler restarts on every deploy. While it is out, schedules are
// unknown: say so, never "no schedules" and never an error. Mount it before
// the reads settle and toggle `show`, so screen readers hear the notice: a
// live region added together with its text is often not announced.
export function SchedulesUnavailableNotice({
  show = true,
  onRetry,
  retrying = false,
  stacked = false,
  className,
}: Props) {
  return (
    <div
      className={
        show
          ? cn(
              "flex flex-col gap-3 rounded-md border border-border-base bg-surface-muted px-3 py-2",
              !stacked && "sm:flex-row sm:items-center sm:justify-between",
              className,
            )
          : "sr-only"
      }
      data-testid={show ? "schedules-unavailable" : undefined}
    >
      <div
        role="status"
        className="flex items-start gap-2"
        data-testid="schedules-unavailable-status"
      >
        {show && (
          <>
            <Icon
              icon={Clock01Icon}
              size={16}
              tone="muted"
              aria-hidden="true"
              className="mt-0.5 shrink-0"
            />
            <Text variant="body" tone="secondary">
              Schedules are temporarily unavailable — your schedules are safe
              and will run once it&apos;s back.
            </Text>
          </>
        )}
      </div>
      {show && onRetry && (
        <Button
          variant="secondary"
          size="small"
          loading={retrying}
          onClick={onRetry}
          className={cn("shrink-0 self-start", !stacked && "sm:self-auto")}
        >
          Try again
        </Button>
      )}
    </div>
  );
}
