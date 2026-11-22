import {
  getGetPageQueryKey,
  useGetPage,
} from "@/app/api/__generated__/endpoints/pages/pages";
import { okData } from "@/app/api/helpers";
import { trackPageOpened } from "@/services/pages/pages-analytics";
import { keepPreviousData, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { hasActiveResult } from "../components/Page/helpers";
import type { PageLive } from "../components/Page/Page";
import {
  useFollowedRunIDs,
  useFollowedRunStore,
} from "../components/Page/store";
import { usePageActions } from "../components/Page/usePageActions";

const POLL_MS = 2_000;

export function usePageView(pageID: string) {
  const runIDs = useFollowedRunIDs(pageID);
  const follow = useFollowedRunStore((s) => s.follow);
  const queryClient = useQueryClient();
  const router = useRouter();

  const query = useGetPage(
    pageID,
    runIDs.length ? { run_id: runIDs } : undefined,
    {
      query: {
        retry: false,
        select: okData,
        // Following a new run must not blank the page and reset its forms.
        placeholderData: keepPreviousData,
        refetchInterval: (q) =>
          hasActiveResult(okData(q.state.data)) ? POLL_MS : false,
      },
    },
  );
  const page = query.data;
  const queryKey = getGetPageQueryKey(pageID);
  const actions = usePageActions({
    page,
    queryKey,
    onDeleted: () => router.push("/pages"),
  });

  const opened = Boolean(page);
  useEffect(() => {
    if (opened) trackPageOpened("pages");
  }, [opened, pageID]);

  const live: PageLive = {
    pageKey: pageID,
    onRunStarted: (libraryAgentID, runID) =>
      follow(pageID, libraryAgentID, runID),
    onReviewed: () => void queryClient.invalidateQueries({ queryKey }),
  };

  return {
    page,
    isLoading: query.isLoading,
    isMissing: !query.isLoading && !page,
    live,
    actions,
  };
}
