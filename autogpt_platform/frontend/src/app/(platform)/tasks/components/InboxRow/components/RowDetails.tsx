import { Text } from "@/components/atoms/Text/Text";
import type { ComponentProps } from "react";
import type { Task } from "../../../data/types";
import type { RowAction } from "../helpers";
import { RowActions } from "./RowActions";
import { StepList } from "./StepList";

interface Props {
  task: Task;
  now: number;
  action: RowAction;
  busy: boolean;
  pending: ComponentProps<typeof RowActions>["pending"];
  onAnswer(option: string): void;
  onApprove(): void;
  onReject(): void;
  onRetry(): void;
  onOpen(): void;
}

export function RowDetails(props: Props) {
  const { task, now, action } = props;
  const question = action === "answer" ? task.open_question : null;
  const outcome = task.inbox.state === "done" ? task.outcome_summary : null;
  return (
    <div className="flex flex-col gap-3">
      <StepList steps={task.inbox.steps} now={now} />
      {question ? (
        <div className="flex flex-col gap-0.5 rounded-control bg-surface-wash px-3 py-2">
          <Text variant="small" tone="waiting" weight="medium">
            {question.asked_by.name} asks
          </Text>
          <Text variant="inbox" tone="ink" className="text-pretty">
            {question.question}
          </Text>
        </div>
      ) : null}
      {outcome ? (
        <Text variant="inbox" tone="text" className="line-clamp-2">
          {outcome}
        </Text>
      ) : null}
      <RowActions {...props} />
    </div>
  );
}
