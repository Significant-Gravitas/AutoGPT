import {
  getGetMyBusinessUnderstandingQueryKey,
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
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useMemoryScope } from "./useMemoryScope";
import { RECENT_FACTS_LIMIT } from "./helpers";

export function useMemoryPage() {
  const scope = useMemoryScope();
  const { scopeExpertID } = scope;
  const queryClient = useQueryClient();

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

  return {
    ...scope,
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
