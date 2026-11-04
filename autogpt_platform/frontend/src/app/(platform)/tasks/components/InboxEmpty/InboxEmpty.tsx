import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { InboxTab } from "../../helpers";
import { NewTaskButton } from "../NewTaskButton/NewTaskButton";
import { EMPTY_COPY, FILTERED_COPY, NEW_USER_COPY } from "./helpers";

interface Props {
  tab: InboxTab;
  isNewUser: boolean;
  hasFilters: boolean;
  onClearFilters: () => void;
  onNewTask: () => void;
}

export function InboxEmpty({
  tab,
  isNewUser,
  hasFilters,
  onClearFilters,
  onNewTask,
}: Props) {
  const copy = hasFilters
    ? FILTERED_COPY
    : isNewUser
      ? NEW_USER_COPY
      : EMPTY_COPY[tab];

  return (
    <div
      key={`${tab}-${hasFilters}-${isNewUser}`}
      className="flex animate-inbox-empty flex-col items-center gap-1 px-6 py-14 text-center"
    >
      <Text variant="inbox-medium" tone="ink">
        {copy.title}
      </Text>
      <Text variant="inbox" tone="quiet" className="max-w-sm text-balance">
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
      ) : isNewUser ? (
        <div className="mt-3">
          <NewTaskButton onClick={onNewTask} />
        </div>
      ) : null}
    </div>
  );
}
