import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  onRetry: () => void;
}

/** The tasks couldn't load: a plain sentence and a way to try again. */
export function InboxError({ onRetry }: Props) {
  return (
    <div
      role="alert"
      className="flex flex-col items-center gap-1 px-6 py-14 text-center"
    >
      <Text variant="inbox-medium" tone="ink">
        Couldn&apos;t load your tasks
      </Text>
      <Text variant="inbox" tone="quiet" className="max-w-xs text-balance">
        Check your connection and try again.
      </Text>
      <Button variant="secondary" size="xs" className="mt-3" onClick={onRetry}>
        Retry
      </Button>
    </div>
  );
}
