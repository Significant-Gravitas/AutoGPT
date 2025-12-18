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

export const dynamic = "force-dynamic";

export type MarketplaceAgentPageParams = { creator: string; slug: string };

export async function generateMetadata({
  params: _params,
}: {
  params: Promise<MarketplaceAgentPageParams>;
}): Promise<Metadata> {
  const params = await _params;
  const { data: creator_agent } = await getV2GetSpecificAgent(
    params.creator,
    params.slug,
  );
  return {
    title: `${(creator_agent as StoreAgentDetails).agent_name} - AutoGPT Marketplace`,
    description: (creator_agent as StoreAgentDetails).description,
  };
}

export default async function MarketplaceAgentPage({
  params: _params,
}: {
  params: Promise<MarketplaceAgentPageParams>;
}) {
  const queryClient = getQueryClient();

  const params = await _params;
  const creator_lower = params.creator.toLowerCase();
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
  if (status === 200 && user && creator_agent.active_version_id) {
    await prefetchGetV2GetAgentByStoreIdQuery(
      queryClient,
      creator_agent.active_version_id,
      {
        query: {
          enabled: true,
        },
      },
    );
  }

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <MainAgentPage params={params} />
    </HydrationBoundary>
  );
}
