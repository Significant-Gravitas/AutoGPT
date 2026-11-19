import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";
import { pollInterval } from "@/app/(platform)/tasks/helpers";
import { useWorkAreaStore } from "../../../../workArea";

/** The API's page size cap: the board shows the newest hundred tasks. */
const LIST_LIMIT = 100;
/** A tab's name until its task is read. */
const UNREAD_TITLE = "Task";

export function useHomeBoard() {
  const openTab = useWorkAreaStore((s) => s.openTask);
  const openTabs = useWorkAreaStore((s) => s.tabs);
  const activeKey = useWorkAreaStore((s) => s.activeKey);
  const isFolded = useWorkAreaStore((s) => s.isFolded);
  const tasksQuery = useListTasks(
    { limit: LIST_LIMIT },
    {
      query: {
        select: (res) => okData(res)?.tasks ?? [],
        refetchInterval: (query) =>
          pollInterval(okData(query.state.data)?.tasks),
      },
    },
  );
  const expertsQuery = useListExperts({
    query: { select: (res) => okData(res) ?? [] },
  });
  const tasks = tasksQuery.data ?? [];
  const front = openTabs.find((tab) => tab.key === activeKey);

  return {
    tasks,
    experts: expertsQuery.data ?? [],
    isLoading: tasksQuery.isLoading,
    isError: tasksQuery.isError && tasksQuery.data === undefined,
    refetch: tasksQuery.refetch,
    /** The task open in the sidebar, whose card is marked. */
    selectedID: !isFolded && front?.kind === "task" ? front.taskID : null,
    // A task opens beside the board, in the same sidebar a chat has.
    openTask(id: string) {
      const title = tasks.find((task) => task.id === id)?.title;
      openTab(id, title ?? UNREAD_TITLE);
    },
  };
}
