"use client";

import { chatHref, ownerOf } from "@/app/(platform)/tasks/helpers";
import { scheduleStateOf } from "@/app/(platform)/tasks/schedule";
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

/** The task's Details block: when it starts if it waits for a start time,
 *  how it repeats, who owns it, the chat doing the work and when it
 *  started. Where it stands leads the tab in its own Status block. */
export function TaskDetails({ task, sessionID, experts }: Props) {
  const owner = ownerOf(task, experts);
  const workChat = workChatOf(task, sessionID);
  const schedule = scheduleStateOf(task);

  return (
    <InfoGroup label="Details">
      <InfoList>
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
