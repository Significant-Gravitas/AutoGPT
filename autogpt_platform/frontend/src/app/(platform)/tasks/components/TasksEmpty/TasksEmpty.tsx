import { Text } from "@/components/atoms/Text/Text";
import type { TaskTab } from "../../helpers";
import { NewTaskButton } from "../NewTaskButton/NewTaskButton";
import { EMPTY_COPY, NEW_USER_COPY } from "./helpers";

interface Props {
  tab: TaskTab;
  isNewUser: boolean;
  onNewTask: () => void;
}

export function TasksEmpty({ tab, isNewUser, onNewTask }: Props) {
  const copy = isNewUser ? NEW_USER_COPY : EMPTY_COPY[tab];

  return (
    <div
      key={`${tab}-${isNewUser}`}
      className="flex animate-inbox-empty flex-col items-center gap-1 px-6 py-14 text-center"
    >
      <Text variant="inbox-medium" tone="ink">
        {copy.title}
      </Text>
      <Text variant="inbox" tone="quiet" className="max-w-sm text-balance">
        {copy.body}
      </Text>
      {isNewUser ? (
        <div className="mt-3">
          <NewTaskButton onClick={onNewTask} />
        </div>
      ) : null}
    </div>
  );
}
