import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  message: string;
  onRetry: () => unknown;
}

/** A card's failed load, said plainly with a way to try again, so it never
 *  reads as an empty history. */
export function CardError({ message, onRetry }: Props) {
  return (
    <div className="flex items-center justify-between gap-3 py-2">
      <Text variant="body" tone="muted">
        {message}
      </Text>
      <Button variant="secondary" size="small" onClick={() => void onRetry()}>
        Retry
      </Button>
    </div>
  );
}
