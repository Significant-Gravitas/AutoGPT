import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { Task } from "../../../data/types";
import { CardApprovals } from "../../CardApprovals/CardApprovals";
import { useInboxRow } from "../../InboxRow/useInboxRow";

interface Props {
  task: Task;
}

/** What a waiting card needs from you: the hire cards, the question with
 *  its suggested answer, or the first way out of what stopped it. */
export function CardExtra({ task }: Props) {
  const row = useInboxRow(task);
  if (row.cards.length) {
    return <CardApprovals taskId={task.id} cards={row.cards} />;
  }
  const text = task.ask?.question ?? task.blocker?.short ?? null;

  return (
    <div className="flex flex-col gap-2">
      {text ? (
        <Text variant="inbox" tone="ink" className="break-words">
          {text}
        </Text>
      ) : null}
      {row.action ? (
        <Button
          variant="primary"
          size="xs"
          className="max-w-full self-start"
          title={row.action.label}
          loading={row.busy}
          onClick={row.runAction}
        >
          <span className="truncate">{row.action.label}</span>
        </Button>
      ) : null}
    </div>
  );
}
