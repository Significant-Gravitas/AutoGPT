"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { formatTimeAgo } from "@/lib/utils/time";
import { useState } from "react";
import { RequestChangesForm } from "./RequestChangesForm";
import { useTaskReview } from "./useTaskReview";

interface Props {
  task: Task;
}

/** Under a finished task's result: Accept it, or Request changes with a
 *  note, which runs the task again. Once accepted, a quiet line says so. */
export function ReviewActions({ task }: Props) {
  const review = useTaskReview(task.id);
  const [isWriting, setIsWriting] = useState(false);

  if (task.review?.state === "accepted") {
    return (
      <Text as="span" variant="small" tone="quiet">
        You accepted this ·{" "}
        {formatTimeAgo(new Date(task.review.at).toISOString())}
      </Text>
    );
  }
  if (task.status !== TaskStatus.DONE || task.repeat) return null;
  if (isWriting) {
    return (
      <RequestChangesForm
        taskID={task.id}
        isPending={review.isRequesting}
        onSubmit={(note) =>
          review.requestChanges(note, () => setIsWriting(false))
        }
        onCancel={() => setIsWriting(false)}
      />
    );
  }
  return (
    <span className="flex items-center gap-2">
      <Button
        variant="primary"
        size="xs"
        loading={review.isAccepting}
        onClick={review.accept}
      >
        Accept
      </Button>
      <Button variant="secondary" size="xs" onClick={() => setIsWriting(true)}>
        Request changes
      </Button>
    </span>
  );
}
