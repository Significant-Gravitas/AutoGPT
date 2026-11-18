import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  unavailable: boolean;
  retrying: boolean;
  onRetry: () => void;
}

// Always mounted, so screen readers hear the notice when it appears: a live
// region added together with its text is often not announced.
export function SchedulesUnavailable({
  unavailable,
  retrying,
  onRetry,
}: Props) {
  return (
    <div
      className={
        unavailable
          ? "flex flex-col gap-3 rounded-md border border-border-base bg-surface-muted px-3 py-2 sm:flex-row sm:items-center sm:justify-between"
          : "sr-only"
      }
      data-testid={unavailable ? "schedules-unavailable" : undefined}
    >
      <div role="status" data-testid="schedules-unavailable-status">
        {unavailable && (
          <Text variant="body" tone="secondary">
            Scheduled items are temporarily unavailable. They&apos;ll be back in
            a moment.
          </Text>
        )}
      </div>
      {unavailable && (
        <Button
          variant="secondary"
          size="small"
          loading={retrying}
          onClick={onRetry}
        >
          Try again
        </Button>
      )}
    </div>
  );
}
