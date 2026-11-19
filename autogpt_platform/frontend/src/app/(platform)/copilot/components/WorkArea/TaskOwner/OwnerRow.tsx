"use client";

import { OwnerPicker } from "@/app/(platform)/tasks/components/NewTaskDialog/components/OwnerPicker/OwnerPicker";
import {
  OTTO_VALUE,
  ownerChoices,
  resolveChoice,
} from "@/app/(platform)/tasks/components/NewTaskDialog/helpers";
import { ownerOf } from "@/app/(platform)/tasks/helpers";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { InfoMeta } from "../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockRow } from "../blocks";
import { useOwnerChange } from "./useOwnerChange";

interface Props {
  task: Task;
  experts: Expert[];
}

/** Who has the task. It can go to Otto or an expert on the team, except
 *  while it is being worked on: then it has to be stopped first. */
export function OwnerRow({ task, experts }: Props) {
  const owner = ownerOf(task, experts);
  const change = useOwnerChange(task);
  const isWorked =
    task.status === TaskStatus.IN_PROGRESS ||
    task.status === TaskStatus.NEEDS_INPUT;
  const value = task.owner_expert_id ?? OTTO_VALUE;
  const active = ownerChoices(experts.filter((e) => !e.is_archived));
  // An owner who left the team still shows as who has it.
  const choices = active.some((choice) => choice.value === value)
    ? active
    : [...active, { value, label: owner.name, avatarUrl: owner.avatarUrl }];

  return (
    <BlockRow
      tile={
        <ExpertAvatar name={owner.name} avatarUrl={owner.avatarUrl} size={26} />
      }
      title="Owner"
      subtitle={isWorked ? "Stop it to change who has it" : null}
      trailing={
        isWorked ? (
          <InfoMeta>{owner.name}</InfoMeta>
        ) : (
          <OwnerPicker
            value={resolveChoice(choices, value)}
            choices={choices}
            onChange={change.pick}
          />
        )
      }
    />
  );
}
