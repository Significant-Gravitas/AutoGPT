import { getServerUser } from "@/lib/supabase/server/getServerUser";
import { MainAgentPage } from "../../../components/MainAgentPage/MainAgentPage";
import {
  getV2GetSpecificAgent,
  prefetchGetV2GetSpecificAgentQuery,
  prefetchGetV2ListStoreAgentsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { prefetchGetV2GetAgentByStoreIdQuery } from "@/app/api/__generated__/endpoints/library/library";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";

// Force dynamic rendering to avoid static generation issues with cookies
export const dynamic = "force-dynamic";

export interface MarketplaceAgentPageParams {
  creator: string;
  slug: string;
}

// TODO : Add generateMetadata here

export default async function MarketplaceAgentPage({
  params: _params,
}: {
  params: Promise<MarketplaceAgentPageParams>;
}) {
  const params = await _params;
  const creator_lower = params.creator.toLowerCase();
  const { user } = await getServerUser();

  const queryClient = getQueryClient();

  const { data } = await getV2GetSpecificAgent(creator_lower, params.slug);

  const agentData = data as StoreAgentDetails;

  await Promise.all([
    prefetchGetV2GetSpecificAgentQuery(queryClient, creator_lower, params.slug),
    prefetchGetV2ListStoreAgentsQuery(queryClient, {
      creator: creator_lower,
    }),
    prefetchGetV2ListStoreAgentsQuery(queryClient, {
      search_query: agentData.slug.replace(/-/g, " "),
    }),

    prefetchGetV2GetAgentByStoreIdQuery(
      queryClient,
      agentData.active_version_id || "",
      {
        query: {
          enabled: !!user,
        },
      },
    ),
  ]);

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <MainAgentPage params={params} />
    </HydrationBoundary>
  );
}
