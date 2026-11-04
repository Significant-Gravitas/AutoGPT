"use client";

import { AgentCheckpoints } from "@/components/atoms/agent-checkpoints";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { useActivityCheckpoints } from "./useActivityCheckpoints";

interface Props {
  taskId: string;
  title: string;
  running: boolean;
}

/** What happened on the task, as a line of checkpoints: who did what, when,
 *  and what each work turn cost. The newest one shimmers while it works. */
export function ActivityCheckpoints({ taskId, title, running }: Props) {
  const activity = useActivityCheckpoints(taskId, running);

  return (
    <section aria-label="Activity" className="flex flex-col gap-2 px-1.5 py-4">
      {activity.isLoading ? (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-2/3" />
        </div>
      ) : activity.checkpoints.length ? (
        <AgentCheckpoints
          title="Activity"
          task={title}
          checkpoints={activity.checkpoints}
        />
      ) : (
        <Text variant="inbox" tone="faint">
          No activity yet.
        </Text>
      )}
      {activity.hasMore ? (
        <Button
          variant="ghost-muted"
          size="xs"
          className="self-start"
          loading={activity.isLoadingMore}
          onClick={activity.loadMore}
        >
          Show earlier activity
        </Button>
      ) : null}
    </section>
  );
}
