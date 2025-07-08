import { useGetV2GetAgentByStoreId } from "@/app/api/__generated__/endpoints/library/library";
import {
  useGetV2GetSpecificAgent,
  useGetV2ListStoreAgents,
} from "@/app/api/__generated__/endpoints/store/store";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { MarketplaceAgentPageParams } from "../../agent/[creator]/[slug]/page";

interface useMainAgentPageProps {
  params: MarketplaceAgentPageParams;
}

export const useMainAgentPage = ({ params }: useMainAgentPageProps) => {
  const creator_lower = params.creator.toLowerCase();
  const { user } = useSupabase();
  const { data: agentData } = useGetV2GetSpecificAgent(
    creator_lower,
    params.slug,
    {
      query: {
        select: (x) => {
          return x.data as StoreAgentDetails;
        },
      },
    },
  );

  const { data: otherAgents } = useGetV2ListStoreAgents(
    {
      creator: creator_lower,
    },
    {
      query: {
        select: (x) => {
          return x.data as StoreAgentsResponse;
        },
      },
    },
  );

  const { data: similarAgents } = useGetV2ListStoreAgents(
    {
      search_query: agentData?.slug.replace(/-/g, " "),
    },
    {
      query: {
        enabled: !!agentData,
        select: (x) => {
          return x.data as StoreAgentsResponse;
        },
      },
    },
  );
  const { data: libraryAgent } = useGetV2GetAgentByStoreId(
    agentData?.store_listing_version_id || "",
    {
      query: {
        enabled: !!user && !!agentData?.store_listing_version_id,
        select: (x) => {
          return x.data as LibraryAgent | null;
        },
      },
    },
  );

  return { libraryAgent, similarAgents, otherAgents, agentData, user };
};
