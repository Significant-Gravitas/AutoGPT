import { useTaskRecord } from "@/app/(platform)/tasks/components/TaskPanel/components/TaskRecordView/useTaskRecord";
import { useGetTask } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { useQueryClient } from "@tanstack/react-query";
import { taskTabKey, useWorkAreaStore } from "../../workArea";
import { seedFromLists } from "./taskSeed";

/** The tab's own placeholder name, before the task's is known. */
const PLACEHOLDER_TITLE = "Task";

/** The task for its tab. The full record can take a few seconds, so the tab
 *  first draws what is known already: the task as a list or the chat's
 *  status line read it, fetched again on its own; failing that, the name
 *  its tab was opened with. */
export function useTaskPanel(taskID: string) {
  const { record } = useTaskRecord(taskID);
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
    },
  });

  const task = record?.task ?? seedQuery.data ?? null;
  const knownTitle =
    tabTitle && tabTitle !== PLACEHOLDER_TITLE ? tabTitle : null;
  return { record, task, title: task?.title ?? knownTitle };
}
