import { Metadata } from "next";
import { getServerUser } from "@/lib/supabase/server/getServerUser";
import {
  getV2GetSpecificAgent,
  prefetchGetV2GetSpecificAgentQuery,
  prefetchGetV2ListStoreAgentsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { MainAgentPage } from "../../../components/MainAgentPage/MainAgentPage";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { prefetchGetV2GetAgentByStoreIdQuery } from "@/app/api/__generated__/endpoints/library/library";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";

// Enable ISR with 10-minute revalidation
export const revalidate = 600; // 10 minutes in seconds

export type MarketplaceAgentPageParams = { creator: string; slug: string };

export async function generateMetadata({
  params: _params,
}: {
  params: Promise<MarketplaceAgentPageParams>;
}): Promise<Metadata> {
  const params = await _params;
  try {
    const { data: creator_agent } = await getV2GetSpecificAgent(
      params.creator,
      params.slug,
    );
    return {
      metadataBase: new URL(process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || 'https://platform.agpt.co'),
      title: `${(creator_agent as StoreAgentDetails).agent_name} - AutoGPT Marketplace`,
      description: (creator_agent as StoreAgentDetails).description,
    };
  } catch (_error) {
    return {
      metadataBase: new URL(process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || 'https://platform.agpt.co'),
      title: `Agent - AutoGPT Marketplace`,
      description: 'View agent details on AutoGPT Marketplace',
    };
  }
}

export default async function MarketplaceAgentPage({
  params: _params,
}: {
  params: Promise<MarketplaceAgentPageParams>;
}) {
  const queryClient = getQueryClient();

  const params = await _params;
  const creator_lower = params.creator.toLowerCase();
  
  try {
    await Promise.all([
      prefetchGetV2GetSpecificAgentQuery(queryClient, creator_lower, params.slug),
      prefetchGetV2ListStoreAgentsQuery(queryClient, {
        creator: creator_lower,
      }),
      prefetchGetV2ListStoreAgentsQuery(queryClient, {
        search_query: params.slug.replace(/-/g, " "),
      }),
    ]);

    const { user } = await getServerUser();
    const { data: creator_agent, status } = await getV2GetSpecificAgent(
      creator_lower,
      params.slug,
    ); // Already cached in above prefetch
    if (status === 200) {
      await prefetchGetV2GetAgentByStoreIdQuery(
        queryClient,
        creator_agent.active_version_id ?? "",
        {
          query: {
            enabled: !!user && !!creator_agent.active_version_id,
          },
        },
      );
    }
  } catch (error) {
    console.error('Failed to prefetch agent data:', error);
  }

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <MainAgentPage params={params} />
    </HydrationBoundary>
  );
}
