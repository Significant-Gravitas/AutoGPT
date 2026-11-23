"use client";

import { StateIcon } from "@/app/(platform)/tasks/components/StateIcon/StateIcon";
import { statusOf } from "@/app/(platform)/tasks/helpers";
import {
  isQueued,
  QUEUED_LINE,
  QUEUED_WORD,
} from "@/app/(platform)/tasks/queue";
import {
  SCHEDULE_WORDS,
  scheduleStateOf,
} from "@/app/(platform)/tasks/schedule";
import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import {
  InfoGroup,
  InfoList,
  InfoMeta,
} from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockRow } from "./blocks";
import { isSettled } from "./taskBlocks";

interface Props {
  task: Task;
}

/** The outcome in one line: what went wrong, the result's first line, or
 *  where the work is now. */
export function outcomeLine(task: Task) {
  if (isQueued(task)) return QUEUED_LINE;
  if (task.status === TaskStatus.CANCELLED) return null;
  const said = isSettled(task.status)
    ? (task.failure_reason ?? task.result)
    : task.status_text;
  return said?.trim().split("\n")[0] || null;
}

/** Where the task stands, first: its state and, in one line, how it ended
 *  or what it is doing now. */
export function TaskStatusBlock({ task }: Props) {
  const schedule = scheduleStateOf(task);
  const queued = isQueued(task);
  const word = queued
    ? QUEUED_WORD
    : schedule
      ? SCHEDULE_WORDS[schedule]
      : statusOf(task.status).word;

  return (
    <InfoGroup label="Status">
      <InfoList>
        <BlockRow
          tile={
            <StateIcon
              status={task.status}
              schedule={schedule}
              queued={queued}
            />
          }
          title={word}
          subtitle={schedule ? null : outcomeLine(task)}
          trailing={
            (task.attempt ?? 1) > 1 ? (
              <InfoMeta>Attempt {task.attempt}</InfoMeta>
            ) : null
          }
        />
      </InfoList>
    </InfoGroup>
  );
}
