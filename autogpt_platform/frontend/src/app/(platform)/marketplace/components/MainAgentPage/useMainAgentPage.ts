import {
  useGetV2GetSpecificAgent,
  useGetV2ListStoreAgents,
} from "@/app/api/__generated__/endpoints/store/store";
import { MarketplaceAgentPageParams } from "../../agent/[creator]/[slug]/page";
import { useGetV2GetAgentByStoreId } from "@/app/api/__generated__/endpoints/library/library";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

export const useMainAgentPage = ({
  params,
}: {
  params: MarketplaceAgentPageParams;
}) => {
  const creator_lower = params.creator.toLowerCase();
  const { user } = useSupabase();
  const {
    data: agent,
    isLoading: isAgentLoading,
    isError: isAgentError,
  } = useGetV2GetSpecificAgent(creator_lower, params.slug);
  const {
    data: otherAgents,
    isLoading: isOtherAgentsLoading,
    isError: isOtherAgentsError,
  } = useGetV2ListStoreAgents(
    { creator: creator_lower },
    {
      query: {
        select: (x) => {
          return x.data as StoreAgentsResponse;
        },
      },
    },
  );
  const {
    data: similarAgents,
    isLoading: isSimilarAgentsLoading,
    isError: isSimilarAgentsError,
  } = useGetV2ListStoreAgents(
    { search_query: params.slug.replace(/-/g, " ") },
    {
      query: {
        select: (x) => {
          return x.data as StoreAgentsResponse;
        },
      },
    },
  );
  const {
    data: libraryAgent,
    isLoading: isLibraryAgentLoading,
    isError: isLibraryAgentError,
  } = useGetV2GetAgentByStoreId(okData(agent)?.active_version_id ?? "", {
    query: {
      select: (x) => {
        return x.data as LibraryAgent;
      },
      enabled: !!user && !!okData(agent)?.active_version_id,
    },
  });

  const isLoading =
    isAgentLoading ||
    isOtherAgentsLoading ||
    isSimilarAgentsLoading ||
    isLibraryAgentLoading;

  const hasError =
    isAgentError ||
    isOtherAgentsError ||
    isSimilarAgentsError ||
    isLibraryAgentError;

  return {
    agent,
    otherAgents,
    similarAgents,
    libraryAgent,
    isLoading,
    hasError,
    user,
  };
};
