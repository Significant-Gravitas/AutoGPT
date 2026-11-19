import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  onReset: () => void;
}

/** What a search or filter that finds nothing says, with a way back. */
export function NoMatches({ onReset }: Props) {
  return (
    <div
      role="status"
      className="flex flex-col items-center gap-2 rounded-tray bg-surface-tray py-10 text-center"
    >
      <Text as="p" variant="inbox-medium" tone="ink">
        No tasks match
      </Text>
      <Text as="p" variant="inbox" tone="faint">
        Try other words, or show every task again.
      </Text>
      <Button type="button" variant="secondary" size="xs" onClick={onReset}>
        Reset filters
      </Button>
    </div>
  );
}
