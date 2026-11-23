"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import type { TaskArtifact } from "@/app/api/__generated__/models/taskArtifact";
import { useEffect } from "react";
import { useWorkAreaStore } from "../../workArea";
import { InfoLoading } from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { TaskAbout } from "./TaskAbout/TaskAbout";
import { TaskDetails } from "./TaskDetails";
import { TaskHeader } from "./TaskHeader";
import { NeedsYouBlock, OutcomeBlocks, TrailBlocks } from "./TaskRecordBlocks";
import { TaskStatusBlock } from "./TaskStatusBlock";
import { useTaskPanel } from "./useTaskPanel";

interface Props {
  taskID: string;
  /** The chat this panel sits beside; none on Home. */
  sessionID: string | null;
  experts: Expert[];
  onOpenArtifact: (artifact: TaskArtifact) => void;
}

/** A task beside the chat, as blocks of short rows, what happened first:
 *  its name and actions, where it stands, what waits on you, the result
 *  and its runs, what it is for (folded once there is a result; near the
 *  top until then), its details, then what happened along the way. What is
 *  known already shows at once; only what waits on the record shows as a
 *  placeholder. */
export function TaskTab({ taskID, sessionID, experts, onOpenArtifact }: Props) {
  const { record, task, title } = useTaskPanel(taskID);
  const nameTask = useWorkAreaStore((s) => s.nameTask);
  const recordTitle = record?.task.title;
  // Once there is an outcome it leads; what the task is for folds away.
  const hasOutcome = !!(task?.result || task?.failure_reason);
  useEffect(() => {
    if (recordTitle) nameTask(taskID, recordTitle);
  }, [taskID, recordTitle, nameTask]);

  return (
    <div
      aria-label={`Task: ${title ?? "loading"}`}
      aria-busy={!record || undefined}
      className="flex flex-col gap-5"
    >
      <TaskHeader task={task} title={title} />
      {task ? <TaskStatusBlock task={task} /> : <InfoLoading />}
      {record ? <NeedsYouBlock asks={record.asks} /> : null}
      {task && !hasOutcome ? <TaskAbout task={task} /> : null}
      {record ? (
        <OutcomeBlocks
          record={record}
          experts={experts}
          onOpenArtifact={onOpenArtifact}
        />
      ) : (
        <InfoLoading />
      )}
      {task && hasOutcome ? <TaskAbout task={task} isFolded /> : null}
      {task ? (
        <TaskDetails task={task} sessionID={sessionID} experts={experts} />
      ) : null}
      {record ? (
        <TrailBlocks
          record={record}
          experts={experts}
          onOpenArtifact={onOpenArtifact}
        />
      ) : null}
    </div>
  );
}
