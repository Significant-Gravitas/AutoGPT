import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { pollInterval } from "@/app/(platform)/tasks/helpers";
import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import type { UIMessage } from "ai";
import { useWorkAreaStore } from "../../../workArea";
import { useCopilotUIStore } from "../../../store";
import { fileItemToArtifactRef } from "../../ContextPanel/components/FilesTab/helpers";
import { useSessionFiles } from "../../ContextPanel/components/FilesTab/useSessionFiles";
import { getPendingQuestions } from "../../QuestionDock/helpers";
import {
  bySoonestStart,
  isScheduled,
  isWatching,
} from "@/app/(platform)/tasks/schedule";
import {
  chatUsageOf,
  chatWorkflowsOf,
  finishedTaskCallsOf,
  isPreviewable,
  previewOf,
  type ChatWorkflow,
} from "./helpers";

const NO_MESSAGES: UIMessage[] = [];
const PREVIEW_CHARS = 1_200;

/** Everything the "This chat" tab shows, gathered from what the chat already
 *  has: its messages, its files and the tasks started from it. */
export function useChatOverview(sessionID: string) {
  const messages = useWorkAreaStore((s) =>
    s.chatSessionID === sessionID ? s.chatMessages : NO_MESSAGES,
  );
  const files = useSessionFiles(sessionID);
  const hasComputer = useCopilotUIStore(
    (s) => s.artifactPanel.computer != null,
  );
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const openWorkflowRun = useWorkAreaStore((s) => s.openWorkflowRun);

  const tasksQuery = useListTasks(
    { session_id: sessionID, limit: 20 },
    {
      query: {
        select: (res) => (res.status === 200 ? res.data.tasks : []),
        refetchInterval: (query) => {
          const res = query.state.data;
          return pollInterval(res?.status === 200 ? res.data.tasks : undefined);
        },
      },
    },
  );
  // A task Otto just started or scheduled shows up without a reload.
  const taskCalls = finishedTaskCallsOf(messages);
  const refetchTasks = tasksQuery.refetch;
  useEffect(() => {
    if (taskCalls > 0) void refetchTasks();
  }, [taskCalls, refetchTasks]);

  const expertsQuery = useListExperts({
    query: { select: (res) => (res.status === 200 ? res.data : []) },
  });

  // The newest text file the chat made stands as the preview card.
  const outputs = files.generated.map((file) => file.item).toReversed();
  const newest = outputs[0] ?? null;
  const previewFile =
    newest && isPreviewable(newest.name, newest.mime_type) ? newest : null;
  const previewUrl = previewFile
    ? fileItemToArtifactRef(previewFile).sourceUrl
    : null;
  const previewQuery = useQuery({
    queryKey: ["chat-overview-preview", previewUrl],
    enabled: previewUrl !== null,
    staleTime: 60_000,
    queryFn: async () => {
      if (!previewUrl) return "";
      const res = await fetch(previewUrl);
      if (!res.ok) throw new Error(`Preview failed: ${res.status}`);
      return (await res.text()).slice(0, PREVIEW_CHARS);
    },
  });

  const usage = chatUsageOf(messages);
  const allTasks = tasksQuery.data ?? [];
  const experts = expertsQuery.data ?? [];

  return {
    pending: getPendingQuestions(messages),
    tasks: allTasks.filter((task) => !isScheduled(task)),
    // Repeating tasks come back here after each run; paused ones go last.
    scheduled: allTasks
      .filter((task) => isScheduled(task) && !isWatching(task))
      .toSorted(bySoonestStart),
    // Tasks started by an event come back here after each run.
    watching: allTasks.filter((task) => isScheduled(task) && isWatching(task)),
    experts,
    outputs,
    preview:
      previewFile && previewQuery.data
        ? {
            file: previewFile,
            ...previewOf(previewQuery.data, previewFile.name),
          }
        : null,
    workflows: chatWorkflowsOf(messages),
    openWorkflow: (workflow: ChatWorkflow) =>
      openWorkflowRun({
        graphID: workflow.graphID,
        executionID: workflow.runIDs.at(-1) ?? null,
        title: workflow.title,
      }),
    uploads: files.uploaded.map((file) => file.item),
    usage,
    teammates: usage.teammateIDs.map(
      (id) => experts.find((expert) => expert.id === id)?.name ?? "An expert",
    ),
    hasComputer,
    openFile: (file: (typeof outputs)[number]) =>
      openArtifact(fileItemToArtifactRef(file)),
  };
}
