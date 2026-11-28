import {
  getGetPinnedPageQueryKey,
  useGetPinnedPage,
} from "@/app/api/__generated__/endpoints/pages/pages";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { trackPageOpened } from "@/services/pages/pages-analytics";
import { keepPreviousData, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { pagePollInterval } from "../Page/helpers";
import type { PageLive } from "../Page/Page";
import { useFollowedRunIDs, useFollowedRunStore } from "../Page/store";
import { usePageActions } from "../Page/usePageActions";

export function usePinnedPageMount(agent: LibraryAgent) {
  const enabled = Boolean(useGetFlag(Flag.PAGES));
  const pageKey = `pinned:${agent.id}`;
  const runIDs = useFollowedRunIDs(pageKey);
  const follow = useFollowedRunStore((s) => s.follow);
  const queryClient = useQueryClient();
  const queryKey = getGetPinnedPageQueryKey(agent.id);

  const query = useGetPinnedPage(
    agent.id,
    runIDs.length ? { run_id: runIDs } : undefined,
    {
      query: {
        enabled,
        retry: false,
        select: okData,
        placeholderData: keepPreviousData,
        refetchInterval: (q) => pagePollInterval(okData(q.state.data)),
        refetchOnWindowFocus: "always",
      },
    },
  );
  const page = query.data;
  // A failed refetch keeps the last page, so an unpin or delete drops it.
  const dropPage = () => void queryClient.resetQueries({ queryKey });
  const actions = usePageActions({
    page,
    queryKey,
    onDeleted: dropPage,
    onUnpinned: dropPage,
  });

  const opened = Boolean(page);
  useEffect(() => {
    if (opened) trackPageOpened("library");
  }, [opened, agent.id]);

  const live: PageLive = {
    pageKey,
    onRunStarted: (libraryAgentID, runID) =>
      follow(pageKey, libraryAgentID, runID),
    onReviewed: () => void queryClient.invalidateQueries({ queryKey }),
  };

  return {
    enabled,
    isLoading: enabled && query.isLoading,
    page,
    live,
    actions,
  };
}
