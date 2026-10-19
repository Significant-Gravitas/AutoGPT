import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Clock01Icon } from "@hugeicons/core-free-icons";

interface Props {
  onRetry?: () => void;
  compact?: boolean;
  className?: string;
}

export function SchedulesUnavailableNotice({
  onRetry,
  compact = false,
  className,
}: Props) {
  return (
    <div
      role="status"
      className={cn(
        "flex items-center gap-3 rounded-large border border-zinc-200 bg-zinc-50 text-left",
        compact ? "px-3 py-2" : "flex-col px-6 py-10 text-center",
        className,
      )}
      data-testid="schedules-unavailable"
    >
      <Icon
        icon={Clock01Icon}
        size={compact ? 18 : 24}
        className="shrink-0 text-zinc-500"
      />
      <div className="flex min-w-0 flex-col gap-1">
        <Text
          variant={compact ? "body-medium" : "h4"}
          className="text-zinc-800"
        >
          Schedules are temporarily unavailable
        </Text>
        <Text variant={compact ? "small" : "body"} className="!text-zinc-500">
          Your schedules are saved. We&apos;ll check again in a moment.
        </Text>
      </div>
      {onRetry ? (
        <Button
          variant="secondary"
          size="small"
          onClick={onRetry}
          className={compact ? "ml-auto shrink-0" : undefined}
        >
          Try again
        </Button>
      ) : null}
    </div>
  );
}
