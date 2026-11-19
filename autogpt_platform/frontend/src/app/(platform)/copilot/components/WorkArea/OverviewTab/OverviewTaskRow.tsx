"use client";

import { StateIcon } from "@/app/(platform)/tasks/components/StateIcon/StateIcon";
import { ownerOf } from "@/app/(platform)/tasks/helpers";
import {
  isScheduled,
  scheduleStateOf,
  waitingLine,
} from "@/app/(platform)/tasks/schedule";
import { useStartCountdown } from "@/app/(platform)/tasks/useStartCountdown";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import { InfoMeta } from "../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockRow, NeedsYouTag } from "../blocks";

interface Props {
  task: Task;
  experts: Expert[];
  onOpen: () => void;
}

/** One task this chat started: where it stands in words, or how soon it
 *  starts while it waits for its start time, and what waits on the user. */
export function OverviewTaskRow({ task, experts, onOpen }: Props) {
  const countdown = useStartCountdown(isScheduled(task) ? task.start_at : null);
  const schedule = scheduleStateOf(task);
  const repeat = isScheduled(task) ? task.repeat : null;
  const subtitle = repeat
    ? waitingLine(repeat, countdown?.label)
    : countdown
      ? `${countdown.label} · ${countdown.exact}`
      : schedule === "missed"
        ? "Missed its start time"
        : (task.status_text ?? ownerOf(task, experts).name);

  return (
    <BlockRow
      tile={<StateIcon status={task.status} schedule={schedule} />}
      title={task.title}
      subtitle={subtitle}
      onClick={onOpen}
      trailing={
        (task.open_ask_count ?? 0) > 0 ? (
          <NeedsYouTag />
        ) : (task.subtask_count ?? 0) > 0 ? (
          <InfoMeta>
            {task.subtasks_done ?? 0}/{task.subtask_count}
          </InfoMeta>
        ) : null
      }
    />
  );
}
