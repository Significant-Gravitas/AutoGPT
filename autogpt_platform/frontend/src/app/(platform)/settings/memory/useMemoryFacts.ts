import {
  useForgetMyExpertMemoryFact,
  useForgetMyMemoryFact,
  useListMyExpertMemoryFactsInfinite,
  useListMyMemoryFactsInfinite,
} from "@/app/api/__generated__/endpoints/memory/memory";
import { toast } from "@/components/molecules/Toast/use-toast";
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
    } catch {
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
    isLoadingMore: factsQuery.isFetchingNextPage,
    loadMore: () => void factsQuery.fetchNextPage(),
    showAll,
    showFewer,
    forgetFact,
    isLoading: factsQuery.isLoading,
    isError: factsQuery.isError,
    error: factsQuery.error,
    refetch: factsQuery.refetch,
  };
}
