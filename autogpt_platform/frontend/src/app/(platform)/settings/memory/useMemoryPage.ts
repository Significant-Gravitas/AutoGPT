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
import { useListExperts } from "@/services/experts/scoped-queries";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  getActiveExperts,
  getScopeName,
  readExpertScopeFromUrl,
  RECENT_FACTS_LIMIT,
} from "./helpers";

export function useMemoryPage() {
  // The Soul drawer's learned-notes panel links here with ?expert=<id>, so
  // the link opens on that expert. Read once — the dropdown owns it after.
  const [scopeExpertID, setScopeExpertID] = useState<string | null>(
    readExpertScopeFromUrl,
  );
  const queryClient = useQueryClient();
  const organizationID = useOrgTeamStore((state) => state.activeOrgID);
  const isLoaded = useOrgTeamStore((state) => state.isLoaded);
  const request = getTenantRequestInit(organizationID, null, isLoaded);

  function scopedKey(key: readonly unknown[]) {
    return ["organization-memory", organizationID, ...key];
  }

  const expertsQuery = useListExperts(undefined, undefined, {
    organizationId: organizationID,
    teamId: null,
    isReady: isLoaded,
  });
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

  const accountFacts = useListMyMemoryFacts(
    { limit: RECENT_FACTS_LIMIT },
    {
      query: {
        queryKey: scopedKey(
          getListMyMemoryFactsQueryKey({ limit: RECENT_FACTS_LIMIT }),
        ),
        enabled: isLoaded && scopeExpertID === null,
      },
      request,
    },
  );
  const expertFacts = useListMyExpertMemoryFacts(
    scopeExpertID ?? "",
    { limit: RECENT_FACTS_LIMIT },
    {
      query: {
        queryKey: scopedKey(
          getListMyExpertMemoryFactsQueryKey(scopeExpertID ?? "", {
            limit: RECENT_FACTS_LIMIT,
          }),
        ),
        enabled: isLoaded && scopeExpertID !== null,
      },
      request,
    },
  );
  const factsQuery = scopeExpertID === null ? accountFacts : expertFacts;
  const facts =
    factsQuery.data?.status === 200 ? factsQuery.data.data.items : [];

  const accountOverview = useGetMyMemoryOverview({
    query: {
      queryKey: scopedKey(getGetMyMemoryOverviewQueryKey()),
      enabled: isLoaded && scopeExpertID === null,
    },
    request,
  });
  const expertOverview = useGetMyExpertMemoryOverview(scopeExpertID ?? "", {
    query: {
      queryKey: scopedKey(
        getGetMyExpertMemoryOverviewQueryKey(scopeExpertID ?? ""),
      ),
      enabled: isLoaded && scopeExpertID !== null,
    },
    request,
  });
  const overviewQuery =
    scopeExpertID === null ? accountOverview : expertOverview;
  const overview =
    overviewQuery.data?.status === 200 ? overviewQuery.data.data : undefined;

  function invalidateScopeQueries() {
    if (scopeExpertID === null) {
      queryClient.invalidateQueries({
        queryKey: scopedKey(getListMyMemoryFactsQueryKey()),
      });
      queryClient.invalidateQueries({
        queryKey: scopedKey(getGetMyMemoryOverviewQueryKey()),
      });
      queryClient.invalidateQueries({
        queryKey: getGetMyBusinessUnderstandingQueryKey(),
      });
      return;
    }
    queryClient.invalidateQueries({
      queryKey: scopedKey(getListMyExpertMemoryFactsQueryKey(scopeExpertID)),
    });
    queryClient.invalidateQueries({
      queryKey: scopedKey(getGetMyExpertMemoryOverviewQueryKey(scopeExpertID)),
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
    request,
    mutation: { onSuccess: onForgotten, onError: onForgetError },
  });
  const forgetExpertFact = useForgetMyExpertMemoryFact({
    request,
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
    request,
    mutation: { onSuccess: onErased, onError: onEraseError },
  });
  const eraseExpert = useEraseMyExpertMemory({
    request,
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
