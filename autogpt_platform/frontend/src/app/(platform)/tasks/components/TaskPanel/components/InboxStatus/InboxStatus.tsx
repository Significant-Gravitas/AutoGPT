"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatDuration } from "../../../../data/helpers";
import type { Task } from "../../../../data/types";
import { formatClock, getInboxStateLabel } from "../../helpers";
import { LiveTimer } from "../LiveTimer/LiveTimer";
import { StepRow } from "../StepRow/StepRow";

interface Props {
  task: Task;
}

const DOT: Record<Task["inbox"]["state"], string> = {
  running: "bg-info animate-pulse motion-reduce:animate-none",
  waiting: "bg-status-waiting",
  done: "bg-status-ok",
  failed: "bg-error",
};

/** The state in one line, then the last few steps, oldest first. */
export function InboxStatus({ task }: Props) {
  const { inbox } = task;
  const state = getInboxStateLabel(task);
  const isRunning = inbox.state === "running";
  const elapsed = isRunning ? (
    <LiveTimer since={task.started_at ?? task.live?.started_at ?? null} />
  ) : task.elapsed_ms ? (
    formatDuration(task.elapsed_ms)
  ) : null;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex min-w-0 items-center gap-2">
        <span
          aria-hidden
          className={cn(
            "size-2 shrink-0 rounded-full",
            inbox.state === "waiting" && !inbox.needs_you
              ? "bg-foreground-faint"
              : DOT[inbox.state],
          )}
        />
        <Text as="span" variant="inbox-medium" tone={state.tone}>
          {state.label}
        </Text>
        {elapsed ? (
          <Text as="span" variant="inbox" tone="faint" tabular>
            · {elapsed}
          </Text>
        ) : null}
        {inbox.last_activity ? (
          <Text
            as="span"
            variant="inbox"
            tone="quiet"
            className="min-w-0 flex-1 truncate"
          >
            ·{" "}
            {inbox.last_activity.actor_name
              ? `${inbox.last_activity.actor_name}: `
              : ""}
            {inbox.last_activity.text}
          </Text>
        ) : null}
      </div>
      {inbox.steps.length ? (
        <ol
          aria-label="Latest steps"
          className="flex flex-col divide-y divide-border-hairline"
        >
          {inbox.steps.map((step, index) => (
            <li
              key={`${step.time}-${index}`}
              className="flex min-h-9 animate-inbox-enter items-center"
            >
              <StepRow
                status={step.status}
                label={step.label}
                trailing={
                  step.status === "running" ? (
                    <LiveTimer since={step.time} />
                  ) : (
                    formatClock(step.time)
                  )
                }
              />
            </li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}
