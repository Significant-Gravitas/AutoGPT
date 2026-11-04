import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { InboxTab } from "../../helpers";
import { EMPTY_COPY, FILTERED_COPY } from "./helpers";

interface Props {
  tab: InboxTab;
  hasFilters: boolean;
  onClearFilters: () => void;
}

export function InboxEmpty({ tab, hasFilters, onClearFilters }: Props) {
  const copy = hasFilters ? FILTERED_COPY : EMPTY_COPY[tab];

  return (
    <div
      key={`${tab}-${hasFilters}`}
      className="flex animate-inbox-empty flex-col items-center gap-1 px-6 py-14 text-center"
    >
      <Text variant="inbox-medium" tone="ink">
        {copy.title}
      </Text>
      <Text variant="inbox" tone="quiet" className="max-w-xs text-balance">
        {copy.body}
      </Text>
      {hasFilters ? (
        <Button
          variant="ghost-muted"
          size="xs"
          className="mt-2"
          onClick={onClearFilters}
        >
          Clear filters
        </Button>
      ) : null}
    </div>
  );
}
