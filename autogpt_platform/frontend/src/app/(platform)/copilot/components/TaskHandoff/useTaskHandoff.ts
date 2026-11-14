import { ownerOf } from "@/app/(platform)/tasks/helpers";
import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetTaskRecord } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { useState } from "react";
import { useCopilotUIStore } from "../../store";
import { useWorkAreaStore } from "../../workArea";
import { duration, secondsBetween } from "../TaskStatusLine/helpers";
import { fileTypeLabel } from "../WorkArea/OverviewTab/helpers";
import { documentOf } from "../WorkArea/taskBlocks";
import { fileSourceURL, splitResult } from "./helpers";

/** A result that landed this recently is played in; an older one, met on
 *  reopening the chat, is simply there. */
const FRESH_MS = 90_000;

/** A finished task as its owner hands it over: who, how long it took, the
 *  result in their words and the document that holds the rest. */
export function useTaskHandoff(taskID: string) {
  const [openedAt] = useState(() => Date.now());
  const recordQuery = useGetTaskRecord(taskID, {
    query: { select: (res) => (res.status === 200 ? res.data : null) },
  });
  const expertsQuery = useListExperts({
    query: { select: (res) => (res.status === 200 ? res.data : []) },
  });
  const openTask = useWorkAreaStore((s) => s.openTask);
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);

  const record = recordQuery.data ?? null;
  const result = record?.task.result;
  if (!record || !result) {
    return { isLoading: recordQuery.isPending, handoff: null };
  }

  const { task } = record;
  const experts = expertsQuery.data ?? [];
  const expert = experts.find((e) => e.id === task.owner_expert_id);
  const seconds = secondsBetween(task.started_at, task.completed_at);
  const document = documentOf(record.artifacts ?? []);
  const fileID = document?.workspace_file_id;
  const finishedAt = task.completed_at
    ? new Date(task.completed_at).getTime()
    : 0;

  return {
    isLoading: false,
    handoff: {
      title: task.title,
      owner: ownerOf(task, experts),
      role: expert?.job_title || expert?.role || null,
      took: seconds === null ? null : duration(seconds),
      ...splitResult(result),
      document:
        document && fileID
          ? { title: document.title, kind: fileTypeLabel(document.title) }
          : null,
      isFresh: openedAt - finishedAt < FRESH_MS,
      openTask: () => openTask(task.id, task.title),
      openDocument: () => {
        if (!document || !fileID) return;
        openArtifact({
          id: fileID,
          title: document.title,
          mimeType: null,
          sourceUrl: fileSourceURL(fileID),
          origin: "agent",
        });
      },
    },
  };
}
