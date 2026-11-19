import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";
import { useExpertSessionPages } from "@/app/(platform)/copilot/useExpertSessionPages";
import {
  flattenSessions,
  useDeletingSessionIds,
  useSessionList,
} from "@/app/(platform)/copilot/useSessionList";
import { pollInterval } from "@/app/(platform)/tasks/helpers";
import { mergeGroupSessions } from "@/components/layout/AppSidebar/components/RecentChats/helpers";
import { buildChatRows, recentChatsFor } from "./helpers";

/** The API's page size cap: the newest hundred tasks, as Home's board reads. */
const TASK_LIST_LIMIT = 100;

/** Every chat of the recipient, a page at a time, as main chats with the
 *  task chats under them. The polled session list keeps the newest chats'
 *  state fresh; the recipient's own pages reach back to the oldest. */
export function useHomeRecentChats(expertId: string | null) {
  const { sessions: newest } = useSessionList();
  const pages = useExpertSessionPages({
    expertId,
    pinnedSeparately: false,
    enabled: true,
  });
  const tasksQuery = useListTasks(
    { limit: TASK_LIST_LIMIT },
    {
      query: {
        select: (res) => okData(res)?.tasks ?? [],
        refetchInterval: (query) =>
          pollInterval(okData(query.state.data)?.tasks),
      },
    },
  );
  const deletingIds = useDeletingSessionIds();
  const sessions = recentChatsFor(
    mergeGroupSessions(
      recentChatsFor(newest, expertId),
      flattenSessions(pages.data),
    ),
    expertId,
  ).filter((session) => !deletingIds.includes(session.id));

  function loadMore() {
    if (!pages.hasNextPage || pages.isFetchingNextPage) return;
    void pages.fetchNextPage();
  }

  return {
    rows: buildChatRows(sessions, tasksQuery.data ?? []),
    hasMore: !!pages.hasNextPage,
    isLoadingMore: pages.isFetchingNextPage,
    loadMore,
  };
}
