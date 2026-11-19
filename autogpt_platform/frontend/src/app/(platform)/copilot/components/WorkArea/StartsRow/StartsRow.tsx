"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import { useStartCountdown } from "@/app/(platform)/tasks/useStartCountdown";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Clock01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { InfoMeta } from "../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockRow } from "../blocks";
import { ScheduleEditor } from "./ScheduleEditor";

interface Props {
  task: Task;
}

/** The Details row of a scheduled task: a live "Starts in …" with the exact
 *  time, why it is held when it is, and Change, which opens a few quick
 *  start times and a date and time to pick. */
export function StartsRow({ task }: Props) {
  const countdown = useStartCountdown(task.start_at);
  const [isEditing, setIsEditing] = useState(false);
  if (!countdown) return null;

  return (
    <>
      <BlockRow
        tile={<Icon icon={Clock01Icon} size={14} tone="info" />}
        title={countdown.label}
        subtitle={task.status_text}
        wrap
        trailing={
          <span className="flex shrink-0 items-center gap-1">
            <InfoMeta>{countdown.exact}</InfoMeta>
            <Button
              variant="ghost-muted"
              size="xs"
              aria-expanded={isEditing}
              onClick={() => setIsEditing(!isEditing)}
            >
              Change
            </Button>
          </span>
        }
      />
      {isEditing ? (
        <li className="px-2 pt-1 pb-2">
          <ScheduleEditor task={task} onDone={() => setIsEditing(false)} />
        </li>
      ) : null}
    </>
  );
}
