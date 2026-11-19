"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import type { TaskArtifact } from "@/app/api/__generated__/models/taskArtifact";
import { useEffect } from "react";
import { useWorkAreaStore } from "../../workArea";
import { InfoLoading } from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { TaskAbout } from "./TaskAbout/TaskAbout";
import { TaskDetails } from "./TaskDetails";
import { TaskHeader } from "./TaskHeader";
import { NeedsYouBlock, TaskRecordBlocks } from "./TaskRecordBlocks";
import { useTaskPanel } from "./useTaskPanel";

interface Props {
  taskID: string;
  /** The chat this panel sits beside; none on Home. */
  sessionID: string | null;
  experts: Expert[];
  onOpenArtifact: (artifact: TaskArtifact) => void;
}

/** A task beside the chat, as blocks of short rows instead of paragraphs:
 *  its name and actions, its details, what it is for, then what its full
 *  record holds. What is known already shows at once; only what waits on
 *  the record shows as a placeholder. */
export function TaskTab({ taskID, sessionID, experts, onOpenArtifact }: Props) {
  const { record, task, title } = useTaskPanel(taskID);
  const nameTask = useWorkAreaStore((s) => s.nameTask);
  const recordTitle = record?.task.title;
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
      {record ? <NeedsYouBlock asks={record.asks} /> : null}
      {task ? (
        <>
          <TaskDetails task={task} sessionID={sessionID} experts={experts} />
          <TaskAbout task={task} />
        </>
      ) : (
        <InfoLoading />
      )}
      {record ? (
        <TaskRecordBlocks
          record={record}
          experts={experts}
          onOpenArtifact={onOpenArtifact}
        />
      ) : (
        <InfoLoading />
      )}
    </div>
  );
}
