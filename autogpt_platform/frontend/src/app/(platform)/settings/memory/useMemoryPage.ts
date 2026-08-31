import {
  getGetMyExpertMemoryOverviewQueryKey,
  getGetMyMemoryOverviewQueryKey,
  getListMyExpertMemoryFactsQueryKey,
  getListMyMemoryFactsQueryKey,
  useEraseMyExpertMemory,
  useEraseMyMemory,
  useForgetMyExpertMemoryFact,
  useForgetMyMemoryFact,
  useGetMyExpertMemoryOverview,
  useGetMyMemoryOverview,
  useListMyExpertMemoryFacts,
  useListMyMemoryFacts,
} from "@/app/api/__generated__/endpoints/memory/memory";
import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { getActiveExperts, getScopeName, RECENT_FACTS_LIMIT } from "./helpers";

export function useMemoryPage() {
  const [scopeExpertID, setScopeExpertID] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Deliberately not gated on HIRE_EXPERTS: expert memory scopes must stay
  // manageable here even if the hiring UI is dark.
  const expertsQuery = useListExperts();
  const expertsSettled = expertsQuery.data?.status === 200;
  const experts = getActiveExperts(
    expertsQuery.data?.status === 200 ? expertsQuery.data.data : undefined,
  );

  // A selected expert can disappear (fired in another tab, roster changed) —
  // fall back to AutoPilot instead of driving queries with a dead scope id.
  useEffect(() => {
    if (!scopeExpertID || !expertsSettled) return;
    if (!experts.some((expert) => expert.id === scopeExpertID)) {
      setScopeExpertID(null);
    }
  }, [scopeExpertID, expertsSettled, experts]);

  const accountFacts = useListMyMemoryFacts(
    { limit: RECENT_FACTS_LIMIT },
    { query: { enabled: scopeExpertID === null } },
  );
  const expertFacts = useListMyExpertMemoryFacts(
    scopeExpertID ?? "",
    { limit: RECENT_FACTS_LIMIT },
    { query: { enabled: scopeExpertID !== null } },
  );
  const factsQuery = scopeExpertID === null ? accountFacts : expertFacts;
  const facts =
    factsQuery.data?.status === 200 ? factsQuery.data.data.items : [];

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
      return;
    }
    queryClient.invalidateQueries({
      queryKey: getListMyExpertMemoryFactsQueryKey(scopeExpertID),
    });
    queryClient.invalidateQueries({
      queryKey: getGetMyExpertMemoryOverviewQueryKey(scopeExpertID),
    });
  }

  function onForgotten() {
    invalidateScopeQueries();
    toast({ title: "Forgotten" });
  }

  function onForgetError() {
    toast({
      title: "Could not forget that memory",
      description: "Please try again.",
      variant: "destructive",
    });
  }

  const forgetAccountFact = useForgetMyMemoryFact({
    mutation: { onSuccess: onForgotten, onError: onForgetError },
  });
  const forgetExpertFact = useForgetMyExpertMemoryFact({
    mutation: { onSuccess: onForgotten, onError: onForgetError },
  });

  async function forgetFact(uuid: string) {
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
      // onError already surfaced the toast; keep the rejection out of the
      // click handler.
    }
  }

  const forgettingUuid = forgetAccountFact.isPending
    ? forgetAccountFact.variables?.factUuid
    : forgetExpertFact.isPending
      ? forgetExpertFact.variables?.factUuid
      : null;

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
    facts,
    isLoadingFacts: factsQuery.isLoading,
    isFactsError: factsQuery.isError,
    factsError: factsQuery.error,
    refetchFacts: factsQuery.refetch,
    memoryCount: overview?.facts ?? null,
    forgetFact,
    forgettingUuid: forgettingUuid ?? null,
    eraseScope,
    isErasing: eraseAccount.isPending || eraseExpert.isPending,
  };
}
