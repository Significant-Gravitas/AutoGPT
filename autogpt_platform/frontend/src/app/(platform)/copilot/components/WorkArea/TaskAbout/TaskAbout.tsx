"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import {
  CheckmarkCircle02Icon,
  Note01Icon,
  Target02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useState } from "react";
import {
  InfoGroup,
  InfoList,
  InfoMeta,
} from "../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockRow } from "../blocks";
import { EditableField } from "./EditableField";
import { useTaskFieldSave, type TaskTextField } from "./useTaskFieldSave";

interface Props {
  task: Task;
  /** Starts as one line, opened on a click: there is a result to lead. */
  isFolded?: boolean;
}

/** About in one line: the first of its words there is. */
function summaryOf(task: Task) {
  const first = [task.description, task.goal, task.done_when].find(
    (text) => text && text.trim(),
  );
  return first?.trim().split("\n")[0] ?? "Add a description or a goal";
}

/** The API's limits: a description may be long, the others are a line. */
const DESCRIPTION_MAX = 10_000;
const LINE_MAX = 300;

const FIELDS: {
  field: TaskTextField;
  label: string;
  placeholder: string;
  icon: typeof Note01Icon;
  maxLength: number;
}[] = [
  {
    field: "description",
    label: "Description",
    placeholder: "Add a description",
    icon: Note01Icon,
    maxLength: DESCRIPTION_MAX,
  },
  {
    field: "goal",
    label: "Goal",
    placeholder: "Add a goal",
    icon: Target02Icon,
    maxLength: LINE_MAX,
  },
  {
    field: "done_when",
    label: "Done when",
    placeholder: "Say when it counts as done",
    icon: CheckmarkCircle02Icon,
    maxLength: LINE_MAX,
  },
];

/** What the task is for, in its own words: the description, the goal and
 *  when it counts as done, each edited where it stands. */
export function TaskAbout({ task, isFolded = false }: Props) {
  const { save } = useTaskFieldSave(task.id);
  const [isOpen, setIsOpen] = useState(!isFolded);

  if (!isOpen) {
    return (
      <InfoGroup label="About">
        <InfoList>
          <BlockRow
            tile={<Icon icon={Note01Icon} size={14} tone="inherit" />}
            title={summaryOf(task)}
            trailing={<InfoMeta>Show</InfoMeta>}
            onClick={() => setIsOpen(true)}
          />
        </InfoList>
      </InfoGroup>
    );
  }

  return (
    <InfoGroup label="About">
      <InfoList>
        {FIELDS.map(({ field, label, placeholder, icon, maxLength }) => (
          <EditableField
            key={field}
            icon={icon}
            label={label}
            value={task[field] ?? ""}
            placeholder={placeholder}
            maxLength={maxLength}
            onSave={(text) => save(field, text)}
          />
        ))}
      </InfoList>
    </InfoGroup>
  );
}
