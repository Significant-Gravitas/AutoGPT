import {
  useForgetMyExpertMemoryFact,
  useForgetMyMemoryFact,
  useListMyExpertMemoryFactsInfinite,
  useListMyMemoryFactsInfinite,
} from "@/app/api/__generated__/endpoints/memory/memory";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useState } from "react";
import {
  FACTS_PAGE_SIZE,
  getFactsTotal,
  getLoadedFacts,
  getNextFactsCursor,
  RECENT_FACTS_LIMIT,
} from "./helpers";

interface Args {
  scopeExpertID: string | null;
  onForgotten: () => void;
}

export function useMemoryFacts({ scopeExpertID, onForgotten }: Args) {
  const [isExpanded, setIsExpanded] = useState(false);
  // Hidden the moment Forget is clicked, and shown again only if the forget
  // fails, so a refetch that lands while it is pending can't bring it back.
  const [forgottenUUIDs, setForgottenUUIDs] = useState<string[]>([]);
  // Another scope opens on its own recent memories.
  const [stateScope, setStateScope] = useState(scopeExpertID);
  if (stateScope !== scopeExpertID) {
    setStateScope(scopeExpertID);
    setIsExpanded(false);
    setForgottenUUIDs([]);
  }

  const accountFacts = useListMyMemoryFactsInfinite(
    { limit: FACTS_PAGE_SIZE },
    {
      query: {
        enabled: scopeExpertID === null,
        getNextPageParam: getNextFactsCursor,
      },
    },
  );
  const expertFacts = useListMyExpertMemoryFactsInfinite(
    scopeExpertID ?? "",
    { limit: FACTS_PAGE_SIZE },
    {
      query: {
        enabled: scopeExpertID !== null,
        getNextPageParam: getNextFactsCursor,
      },
    },
  );
  const factsQuery = scopeExpertID === null ? accountFacts : expertFacts;
  const pages = factsQuery.data?.pages;

  const loadedFacts = getLoadedFacts(pages);
  const visibleFacts = loadedFacts.filter(
    (fact) => !forgottenUUIDs.includes(fact.uuid),
  );
  const total = getFactsTotal(pages);
  const shownTotal =
    total === null
      ? null
      : Math.max(0, total - (loadedFacts.length - visibleFacts.length));

  const forgetAccountFact = useForgetMyMemoryFact();
  const forgetExpertFact = useForgetMyExpertMemoryFact();

  async function forgetFact(uuid: string) {
    setForgottenUUIDs((current) => [...current, uuid]);
    try {
      if (scopeExpertID === null) {
        await forgetAccountFact.mutateAsync({ factUuid: uuid });
      } else {
        await forgetExpertFact.mutateAsync({
          expertId: scopeExpertID,
          factUuid: uuid,
        });
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        onForgotten();
        toast({ title: "Already forgotten" });
        return;
      }
      // Not refetched: a 500 can be a forget whose clean-up failed, already
      // out of the list, and the row stays so that trying again finishes it.
      setForgottenUUIDs((current) => current.filter((id) => id !== uuid));
      toast({
        title: "Could not forget that memory",
        description: "Please try again.",
        variant: "destructive",
      });
      return;
    }
    onForgotten();
    toast({ title: "Forgotten" });
  }

  function loadMore() {
    void factsQuery.fetchNextPage().then((result) => {
      if (!result.isError) return;
      toast({
        title: "Could not load more memories",
        description: "Please try again.",
        variant: "destructive",
      });
    });
  }

  function showAll() {
    setIsExpanded(true);
  }

  function showFewer() {
    setIsExpanded(false);
  }

  return {
    facts: isExpanded
      ? visibleFacts
      : visibleFacts.slice(0, RECENT_FACTS_LIMIT),
    total: shownTotal,
    isExpanded,
    canShowAll:
      !isExpanded &&
      (visibleFacts.length > RECENT_FACTS_LIMIT || factsQuery.hasNextPage),
    hasMore: isExpanded && factsQuery.hasNextPage,
    isAllLoaded: !factsQuery.hasNextPage,
    isLoadingMore: factsQuery.isFetchingNextPage,
    loadMore,
    showAll,
    showFewer,
    forgetFact,
    isLoading: factsQuery.isLoading,
    // A failed next page or background refetch keeps what is already shown.
    isError: factsQuery.isError && !pages,
    error: factsQuery.error,
    refetch: factsQuery.refetch,
  };
}
