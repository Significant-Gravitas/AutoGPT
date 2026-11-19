"use client";

import { StateIcon } from "@/app/(platform)/tasks/components/StateIcon/StateIcon";
import { chatHref, ownerOf, statusOf } from "@/app/(platform)/tasks/helpers";
import {
  SCHEDULE_WORDS,
  scheduleStateOf,
} from "@/app/(platform)/tasks/schedule";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { formatTimeAgo } from "@/lib/utils/time";
import {
  Calendar03Icon,
  Message01Icon,
  Target02Icon,
} from "@hugeicons/core-free-icons";
import {
  InfoGroup,
  InfoList,
  InfoMeta,
} from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { possessive } from "../ToolChain/expertHandoff";
import { BlockRow } from "./blocks";
import { RepeatRow } from "./StartsRow/RepeatRow";
import { StartsRow } from "./StartsRow/StartsRow";
import { isSettled, workChatOf } from "./taskBlocks";

interface Props {
  task: Task;
  /** The chat the task tab sits beside; none on Home. */
  sessionID: string | null;
  experts: Expert[];
}

/** The task's Details block: where it stands, when it starts if it waits
 *  for a start time, who owns it, the chat doing the work, when it started
 *  and its goal. */
export function TaskDetails({ task, sessionID, experts }: Props) {
  const status = statusOf(task.status);
  const owner = ownerOf(task, experts);
  const workChat = workChatOf(task, sessionID);
  const schedule = scheduleStateOf(task);

  return (
    <InfoGroup label="Details">
      <InfoList>
        <BlockRow
          tile={<StateIcon status={task.status} schedule={schedule} />}
          title="Status"
          subtitle={
            isSettled(task.status) || schedule ? null : task.status_text
          }
          trailing={
            <InfoMeta>
              {schedule ? SCHEDULE_WORDS[schedule] : status.word}
              {(task.attempt ?? 1) > 1 ? ` · attempt ${task.attempt}` : ""}
            </InfoMeta>
          }
        />
        {schedule === "scheduled" ? <StartsRow task={task} /> : null}
        {task.repeat ? <RepeatRow task={task} repeat={task.repeat} /> : null}
        <BlockRow
          tile={
            <ExpertAvatar
              name={owner.name}
              avatarUrl={owner.avatarUrl}
              size={26}
            />
          }
          title="Owner"
          trailing={<InfoMeta>{owner.name}</InfoMeta>}
        />
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
        {task.goal ? (
          <BlockRow
            tile={<Icon icon={Target02Icon} size={14} tone="inherit" />}
            title="Goal"
            subtitle={task.goal}
            wrap
          />
        ) : null}
      </InfoList>
    </InfoGroup>
  );
}
