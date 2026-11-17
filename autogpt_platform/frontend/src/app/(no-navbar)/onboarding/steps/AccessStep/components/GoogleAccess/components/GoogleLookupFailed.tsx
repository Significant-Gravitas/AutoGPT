import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  onRetry: () => void;
  onSkip: () => void;
}

export function GoogleLookupFailed({ onRetry, onSkip }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" role="alert">
        We couldn&apos;t check your Google connection.
      </Text>
      <Button variant="secondary" onClick={onRetry}>
        Try again
      </Button>
      <Button variant="ghost" onClick={onSkip}>
        Not now
      </Button>
    </div>
  );
}
