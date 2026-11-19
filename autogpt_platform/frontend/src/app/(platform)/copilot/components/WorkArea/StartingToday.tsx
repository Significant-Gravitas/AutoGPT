"use client";

import { ownerOf } from "@/app/(platform)/tasks/helpers";
import { formatStartTime } from "@/app/(platform)/tasks/schedule";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Clock01Icon, RepeatIcon } from "@hugeicons/core-free-icons";
import { InfoMeta } from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockRow } from "./blocks";
import { Block, STARTING } from "./BriefParts";
import { useStartingToday } from "./useStartingToday";

interface Props {
  now: number;
  experts: Expert[];
  onOpen: (task: Task) => void;
}

/** The brief's "Starting today": tasks scheduled to start before the day
 *  ends, each with its start time; nothing when there are none. */
export function StartingToday({ now, experts, onOpen }: Props) {
  const tasks = useStartingToday(now);

  return (
    <Block id={STARTING} label="Starting today" count={tasks.length}>
      {tasks.map((task) => (
        <BlockRow
          key={task.id}
          tile={
            <Icon
              icon={task.repeat ? RepeatIcon : Clock01Icon}
              size={14}
              tone="info"
            />
          }
          title={task.title}
          subtitle={
            task.status_text ??
            task.repeat?.label ??
            ownerOf(task, experts).name
          }
          trailing={<InfoMeta>{formatStartTime(task.start_at, now)}</InfoMeta>}
          onClick={() => onOpen(task)}
        />
      ))}
    </Block>
  );
}
