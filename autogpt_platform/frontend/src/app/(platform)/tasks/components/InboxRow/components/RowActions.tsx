import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { RepeatIcon } from "@hugeicons/core-free-icons";
import type { Task } from "../../../data/types";
import type { RowAction } from "../helpers";

interface Props {
  task: Task;
  action: RowAction;
  busy: boolean;
  pending: {
    answer?: string;
    approve: boolean;
    reject: boolean;
    retry: boolean;
  };
  onAnswer(option: string): void;
  onApprove(): void;
  onReject(): void;
  onRetry(): void;
  onOpen(): void;
}

export function RowActions(props: Props) {
  const { task, action, busy, pending, onOpen } = props;
  const options = task.open_question?.options ?? [];
  return (
    <div className="flex flex-wrap items-center gap-2">
      {action === "answer"
        ? options.map((option) => (
            <Button
              key={option}
              variant="secondary"
              size="small"
              disabled={busy}
              loading={pending.answer === option}
              onClick={() => props.onAnswer(option)}
            >
              {option}
            </Button>
          ))
        : null}
      {action === "review" ? (
        <>
          <Button
            variant="primary"
            size="small"
            disabled={busy}
            loading={pending.approve}
            onClick={props.onApprove}
            aria-keyshortcuts="A"
          >
            Approve
          </Button>
          <Button
            variant="secondary"
            size="small"
            disabled={busy}
            loading={pending.reject}
            onClick={props.onReject}
          >
            Reject
          </Button>
        </>
      ) : null}
      {action === "retry" ? (
        <Button
          variant="primary"
          size="small"
          leadingIcon={RepeatIcon}
          disabled={busy}
          loading={pending.retry}
          onClick={props.onRetry}
        >
          Retry
        </Button>
      ) : null}
      <Button
        variant={action === "open" ? "secondary" : "ghost"}
        size="small"
        onClick={onOpen}
      >
        Open
      </Button>
      {action === "review" ? (
        <Text as="span" variant="small" tone="faint">
          Press A to approve
        </Text>
      ) : null}
    </div>
  );
}
