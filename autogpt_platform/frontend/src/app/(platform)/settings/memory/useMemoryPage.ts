import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import {
  getGetMyBusinessUnderstandingQueryKey,
  getGetMyExpertMemoryOverviewQueryKey,
  getGetMyMemoryOverviewQueryKey,
  getListMyExpertMemoryFactsQueryKey,
  getListMyMemoryFactsQueryKey,
  useEraseMyExpertMemory,
  useEraseMyMemory,
  useGetMyExpertMemoryOverview,
  useGetMyMemoryOverview,
} from "@/app/api/__generated__/endpoints/memory/memory";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import {
  getActiveExperts,
  getScopeName,
  readExpertScopeFromUrl,
} from "./helpers";
import { useMemoryFacts } from "./useMemoryFacts";

export function useMemoryPage() {
  // The Soul drawer's learned-notes panel links here with ?expert=<id>, so
  // the link opens on that expert. Read once — the dropdown owns it after.
  const [scopeExpertID, setScopeExpertID] = useState<string | null>(
    readExpertScopeFromUrl,
  );
  const queryClient = useQueryClient();

  const expertsQuery = useListExperts();
  const expertsSettled = expertsQuery.data?.status === 200;
  const experts = getActiveExperts(
    expertsQuery.data?.status === 200 ? expertsQuery.data.data : undefined,
  );

  // A selected expert can disappear (fired in another tab, roster changed) —
  // fall back to Otto instead of driving queries with a dead scope id.
  useEffect(() => {
    if (!scopeExpertID || !expertsSettled) return;
    if (!experts.some((expert) => expert.id === scopeExpertID)) {
      setScopeExpertID(null);
    }
  }, [scopeExpertID, expertsSettled, experts]);

  const memories = useMemoryFacts({
    scopeExpertID,
    onForgotten: invalidateScopeQueries,
  });
  const [isEraseOpen, setIsEraseOpen] = useState(false);

  const accountOverview = useGetMyMemoryOverview({
    query: { enabled: scopeExpertID === null },
  });
  const expertOverview = useGetMyExpertMemoryOverview(scopeExpertID ?? "", {
    query: { enabled: scopeExpertID !== null },
  });
  const overviewQuery =
    scopeExpertID === null ? accountOverview : expertOverview;
  const overview =
    overviewQuery.data?.status === 200 ? overviewQuery.data.data : undefined;

  function invalidateScopeQueries() {
    if (scopeExpertID === null) {
      queryClient.invalidateQueries({
        queryKey: getListMyMemoryFactsQueryKey(),
      });
      queryClient.invalidateQueries({
        queryKey: getGetMyMemoryOverviewQueryKey(),
      });
      queryClient.invalidateQueries({
        queryKey: getGetMyBusinessUnderstandingQueryKey(),
      });
      return;
    }
    queryClient.invalidateQueries({
      queryKey: getListMyExpertMemoryFactsQueryKey(scopeExpertID),
    });
    queryClient.invalidateQueries({
      queryKey: getGetMyExpertMemoryOverviewQueryKey(scopeExpertID),
    });
  }

  function onErased() {
    invalidateScopeQueries();
    toast({ title: "Memory erased" });
  }

  function onEraseError() {
    toast({
      title: "Could not erase memory",
      description: "Please try again.",
      variant: "destructive",
    });
  }

  const eraseAccount = useEraseMyMemory({
    mutation: { onSuccess: onErased, onError: onEraseError },
  });
  const eraseExpert = useEraseMyExpertMemory({
    mutation: { onSuccess: onErased, onError: onEraseError },
  });

  async function eraseScope() {
    try {
      if (scopeExpertID === null) {
        await eraseAccount.mutateAsync();
      } else {
        await eraseExpert.mutateAsync({ expertId: scopeExpertID });
      }
      return true;
    } catch {
      // onError already surfaced the toast; the caller keeps its dialog open.
      return false;
    }
  }

  function selectScope(value: string | null) {
    setScopeExpertID(value);
  }

  return {
    scopeExpertID,
    selectScope,
    experts,
    scopeName: getScopeName(scopeExpertID, experts),
    memories,
    memoryCount: overview?.facts ?? null,
    isEraseOpen,
    setIsEraseOpen,
    eraseScope,
    isErasing: eraseAccount.isPending || eraseExpert.isPending,
  };
}
