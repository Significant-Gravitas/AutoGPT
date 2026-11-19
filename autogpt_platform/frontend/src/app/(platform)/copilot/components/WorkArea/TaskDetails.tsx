"use client";

import { StateIcon } from "@/app/(platform)/tasks/components/StateIcon/StateIcon";
import { chatHref, ownerOf, statusOf } from "@/app/(platform)/tasks/helpers";
import {
  isQueued,
  QUEUED_LINE,
  QUEUED_WORD,
} from "@/app/(platform)/tasks/queue";
import {
  SCHEDULE_WORDS,
  scheduleStateOf,
} from "@/app/(platform)/tasks/schedule";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import { Icon } from "@/components/atoms/Icon/Icon";
import { formatTimeAgo } from "@/lib/utils/time";
import { Calendar03Icon, Message01Icon } from "@hugeicons/core-free-icons";
import {
  InfoGroup,
  InfoList,
  InfoMeta,
} from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { possessive } from "../ToolChain/expertHandoff";
import { BlockRow } from "./blocks";
import { RepeatRow } from "./StartsRow/RepeatRow";
import { StartsRow } from "./StartsRow/StartsRow";
import { OwnerRow } from "./TaskOwner/OwnerRow";
import { isSettled, workChatOf } from "./taskBlocks";

interface Props {
  task: Task;
  /** The chat the task tab sits beside; none on Home. */
  sessionID: string | null;
  experts: Expert[];
}

/** The task's Details block: where it stands, when it starts if it waits
 *  for a start time, who owns it, the chat doing the work and when it
 *  started. Its goal sits in the About block, where it can be edited. */
export function TaskDetails({ task, sessionID, experts }: Props) {
  const status = statusOf(task.status);
  const owner = ownerOf(task, experts);
  const workChat = workChatOf(task, sessionID);
  const schedule = scheduleStateOf(task);
  // A run waiting for a free turn slot has not started working yet.
  const queued = isQueued(task);

  return (
    <InfoGroup label="Details">
      <InfoList>
        <BlockRow
          tile={
            <StateIcon
              status={task.status}
              schedule={schedule}
              queued={queued}
            />
          }
          title="Status"
          subtitle={
            queued
              ? QUEUED_LINE
              : isSettled(task.status) || schedule
                ? null
                : task.status_text
          }
          trailing={
            <InfoMeta>
              {queued
                ? QUEUED_WORD
                : schedule
                  ? SCHEDULE_WORDS[schedule]
                  : status.word}
              {(task.attempt ?? 1) > 1 ? ` · attempt ${task.attempt}` : ""}
            </InfoMeta>
          }
        />
        {schedule === "scheduled" ? <StartsRow task={task} /> : null}
        {task.repeat ? <RepeatRow task={task} repeat={task.repeat} /> : null}
        <OwnerRow task={task} experts={experts} />
        {workChat ? (
          <BlockRow
            tile={<Icon icon={Message01Icon} size={14} tone="inherit" />}
            title={`${possessive(owner.name)} chat`}
            subtitle={
              isSettled(task.status)
                ? "Where the work was done"
                : "Where the work is being done"
            }
            trailing={<InfoMeta>Open</InfoMeta>}
            href={chatHref(workChat)}
          />
        ) : null}
        {task.started_at ? (
          <BlockRow
            tile={<Icon icon={Calendar03Icon} size={14} tone="inherit" />}
            title="Started"
            trailing={
              <InfoMeta>
                {formatTimeAgo(new Date(task.started_at).toISOString())}
              </InfoMeta>
            }
          />
        ) : null}
      </InfoList>
    </InfoGroup>
  );
}
