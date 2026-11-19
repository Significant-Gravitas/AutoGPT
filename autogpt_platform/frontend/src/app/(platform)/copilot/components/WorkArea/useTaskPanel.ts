import { isSettled, POLL_MS } from "@/app/(platform)/tasks/helpers";
import { isTaskGone } from "@/app/(platform)/tasks/taskErrors";
import {
  useGetTask,
  useGetTaskRecord,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { taskTabKey, useWorkAreaStore } from "../../workArea";
import { announceTaskGone } from "./taskErrors";
import { seedFromLists } from "./taskSeed";

/** The tab's own placeholder name, before the task's is known. */
const PLACEHOLDER_TITLE = "Task";
const MAX_RETRIES = 3;

/** Retry a failed read, but not one that says the task is gone. */
function retryUnlessGone(count: number, error: unknown) {
  return !isTaskGone(error) && count < MAX_RETRIES;
}

/** The task for its tab, re-read while it moves and whenever the window
 *  comes back into focus (another tab may have changed it). The full record
 *  can take a few seconds, so the tab first draws what is known already:
 *  the task as a list or the chat's status line read it, fetched again on
 *  its own; failing that, the name its tab was opened with. A task that is
 *  gone closes its tab with one message. */
export function useTaskPanel(taskID: string) {
  const recordQuery = useGetTaskRecord(taskID, {
    query: {
      select: (res) => (res.status === 200 ? res.data : null),
      refetchOnWindowFocus: "always",
      retry: retryUnlessGone,
      refetchInterval: (q) => {
        const res = q.state.data;
        if (res?.status !== 200) return false;
        return isSettled(res.data.task.status) ? false : POLL_MS;
      },
    },
  });
  const record = recordQuery.data ?? null;
  const queryClient = useQueryClient();
  const tabTitle = useWorkAreaStore((s) => {
    const tab = s.tabs.find((item) => item.key === taskTabKey(taskID));
    return tab?.kind === "task" ? tab.title : null;
  });
  const seedQuery = useGetTask(taskID, {
    query: {
      enabled: !record,
      select: (res) => (res.status === 200 ? res.data : null),
      placeholderData: () => seedFromLists(queryClient, taskID),
      refetchOnWindowFocus: "always",
      retry: retryUnlessGone,
    },
  });

  const isGone = isTaskGone(recordQuery.error) || isTaskGone(seedQuery.error);
  useEffect(() => {
    if (isGone) announceTaskGone(taskID);
  }, [isGone, taskID]);

  const task = record?.task ?? seedQuery.data ?? null;
  const knownTitle =
    tabTitle && tabTitle !== PLACEHOLDER_TITLE ? tabTitle : null;
  return { record, task, title: task?.title ?? knownTitle };
}
