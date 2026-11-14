import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import { okData } from "@/app/api/helpers";
import { useHomePage } from "@/app/(platform)/home/useHomePage";
import {
  BOARD_HREF,
  pollInterval,
  taskHref,
} from "@/app/(platform)/tasks/helpers";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useIsMobile } from "../../../../useIsMobile";
import { useWorkAreaStore } from "../../../../workArea";
import { sortToday } from "./helpers";

/** The API's page size cap: the newest hundred tasks. */
const LIST_LIMIT = 100;

/** The day, for the brief: the tasks that need the user, the ones ready to
 *  read and in progress, what is scheduled next and what Otto wrote. */
export function useHomeToday() {
  const [now] = useState(() => Date.now());
  const router = useRouter();
  const isMobile = useIsMobile();
  const openTab = useWorkAreaStore((s) => s.openTask);
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
  const { dashboard } = useHomePage();
  const tasks = tasksQuery.data ?? [];

  return {
    now,
    ...sortToday(tasks, now),
    upcoming: (dashboard?.upcoming_tasks ?? []).toSorted(
      (a, b) =>
        new Date(a.next_run_time).getTime() -
        new Date(b.next_run_time).getTime(),
    ),
    /** The brief Otto wrote this morning, when there is one. */
    briefing: dashboard?.briefing ?? null,
    experts: expertsQuery.data ?? [],
    // Beside Home, in the sidebar a chat has; a phone has no room for one,
    // so there the task opens in its chat.
    openTask(task: Task) {
      if (isMobile) router.push(taskHref(task) ?? BOARD_HREF);
      else openTab(task.id, task.title);
    },
  };
}
