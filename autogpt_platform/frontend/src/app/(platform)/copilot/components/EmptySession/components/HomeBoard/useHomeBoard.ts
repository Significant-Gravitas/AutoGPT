import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";
import { pollInterval } from "@/app/(platform)/tasks/helpers";
import { useRouter } from "next/navigation";
import { taskHref } from "../../../../tools/CreateTask/helpers";

/** The API's page size cap: the board shows the newest hundred tasks. */
const LIST_LIMIT = 100;

export function useHomeBoard() {
  const router = useRouter();
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

  return {
    tasks: tasksQuery.data ?? [],
    experts: expertsQuery.data ?? [],
    isLoading: tasksQuery.isLoading,
    isError: tasksQuery.isError && tasksQuery.data === undefined,
    refetch: tasksQuery.refetch,
    // A task opens in its sheet on the tasks page.
    openTask(id: string) {
      router.push(taskHref(id));
    },
  };
}
