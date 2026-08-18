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
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { getActiveExperts, getScopeName, RECENT_FACTS_LIMIT } from "./helpers";

export function useMemoryPage() {
  const [scopeExpertID, setScopeExpertID] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);

  const expertsQuery = useListExperts({
    query: { enabled: isExpertsEnabled === true },
  });
  const experts = getActiveExperts(
    expertsQuery.data?.status === 200 ? expertsQuery.data.data : undefined,
  );

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
    if (scopeExpertID === null) {
      await forgetAccountFact.mutateAsync({ factUuid: uuid });
      return;
    }
    await forgetExpertFact.mutateAsync({
      expertId: scopeExpertID,
      factUuid: uuid,
    });
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
    if (scopeExpertID === null) {
      await eraseAccount.mutateAsync();
      return;
    }
    await eraseExpert.mutateAsync({ expertId: scopeExpertID });
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
