import {
  useGetV2GetSpecificAgent,
  useGetV2ListStoreAgents,
} from "@/app/api/__generated__/endpoints/store/store";
import type { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import type { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import type { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useState } from "react";

interface Args {
  onPick: (job: { id: string; name: string }) => void;
}

export function useFirstJobStep({ onPick }: Args) {
  const { isLoggedIn } = useAuth();
  const [selected, setSelected] = useState<StoreAgent | null>(null);

  const agentsQuery = useGetV2ListStoreAgents(
    { sorted_by: "runs", page_size: 3 },
    {
      query: {
        select: (x) => x.data as StoreAgentsResponse,
        enabled: isLoggedIn,
      },
    },
  );

  const detailQuery = useGetV2GetSpecificAgent(
    selected?.creator ?? "",
    selected?.slug ?? "",
    undefined,
    {
      query: {
        select: (x) => x.data as StoreAgentDetails,
        enabled: Boolean(selected),
      },
    },
  );

  const versionId = detailQuery.data?.store_listing_version_id ?? null;

  function confirm() {
    if (selected && versionId) {
      onPick({ id: versionId, name: selected.agent_name });
    }
  }

  function retry() {
    if (detailQuery.isError) {
      void detailQuery.refetch();
      return;
    }
    void agentsQuery.refetch();
  }

  return {
    suggestions: agentsQuery.data?.agents ?? [],
    isLoading: isLoggedIn && agentsQuery.isLoading,
    hasError: agentsQuery.isError || detailQuery.isError,
    selected,
    select: setSelected,
    isResolving: Boolean(selected) && detailQuery.isFetching,
    canConfirm: Boolean(selected) && Boolean(versionId),
    confirm,
    retry,
  };
}
