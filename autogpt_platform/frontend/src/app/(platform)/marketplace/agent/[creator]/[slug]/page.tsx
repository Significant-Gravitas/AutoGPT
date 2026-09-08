import { prefetchGetV2GetAgentByStoreIdQuery } from "@/app/api/__generated__/endpoints/library/library";
import {
  getV2GetSpecificAgent,
  prefetchGetV2GetSpecificAgentQuery,
  prefetchGetV2ListStoreAgentsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { getServerUser } from "@/lib/auth/server/getServerUser";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { Metadata } from "next";
import { notFound } from "next/navigation";
import { MainAgentPage } from "../../../components/MainAgentPage/MainAgentPage";

export const dynamic = "force-dynamic";

export type MarketplaceAgentPageParams = { creator: string; slug: string };

export async function generateMetadata({
  params: _params,
}: {
  params: Promise<MarketplaceAgentPageParams>;
}): Promise<Metadata> {
  const params = await _params;

  let creator_agent: StoreAgentDetails;
  try {
    const { data } = await getV2GetSpecificAgent(params.creator, params.slug);
    creator_agent = data as StoreAgentDetails;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }

  return {
    title: `${creator_agent.agent_name} - AutoGPT Marketplace`,
    description: creator_agent.description,
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

  let agentData: StoreAgentDetails | undefined;
  try {
    const { data, status } = await getV2GetSpecificAgent(
      creator_lower,
      params.slug,
    ); // Already cached in above prefetch
    if (status === 200) agentData = data as StoreAgentDetails;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }

  if (user && agentData?.active_version_id) {
    await prefetchGetV2GetAgentByStoreIdQuery(
      queryClient,
      agentData.active_version_id,
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
