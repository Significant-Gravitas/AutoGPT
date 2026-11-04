"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useCancelTask } from "./useCancelTask";

interface Props {
  taskId: string;
}

/** Cancel a task that is working or waiting on you, after one confirm:
 *  it stops the expert's work, so a stray click shouldn't do it. */
export function CancelTask({ taskId }: Props) {
  const cancel = useCancelTask(taskId);

  if (!cancel.isConfirming) {
    return (
      <Button variant="ghost-muted" size="xs" onClick={cancel.ask}>
        Cancel task
      </Button>
    );
  }

  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <Text as="span" variant="small" tone="quiet">
        Stop this task?
      </Text>
      <Button
        variant="primary"
        size="xs"
        loading={cancel.isCancelling}
        onClick={cancel.confirm}
      >
        Yes, cancel
      </Button>
      <Button
        variant="ghost-muted"
        size="xs"
        disabled={cancel.isCancelling}
        onClick={cancel.keep}
      >
        Keep it
      </Button>
    </span>
  );
}
