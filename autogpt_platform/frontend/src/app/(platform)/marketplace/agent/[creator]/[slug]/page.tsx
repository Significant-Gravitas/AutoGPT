import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { notFound } from "next/navigation";
import { prefetchGetV2GetAgentByStoreIdQuery } from "@/app/api/__generated__/endpoints/library/library";
import {
  getV2GetSpecificAgent,
  prefetchGetV2GetSpecificAgentQuery,
  prefetchGetV2ListStoreAgentsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { getClientIPToken } from "@/lib/auth/client-ip-token";
import { getServerUser } from "@/lib/auth/server/getServerUser";
import { CLIENT_IP_TOKEN_HEADER_NAME } from "@/lib/constants";
import { buildPageMetadata } from "@/lib/metadata";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { Metadata } from "next";
import { headers } from "next/headers";
import { MainAgentPage } from "../../../components/MainAgentPage/MainAgentPage";

export const dynamic = "force-dynamic";

export type MarketplaceAgentPageParams = { creator: string; slug: string };

export async function generateMetadata({
  params: _params,
}: {
  params: Promise<MarketplaceAgentPageParams>;
}): Promise<Metadata> {
  const params = await _params;
  const { data } = await getAgentOrNotFound(params.creator, params.slug);
  const agent = data as StoreAgentDetails;

  return buildPageMetadata({
    title: `${agent.agent_name} - AutoGPT Marketplace`,
    description: agent.sub_heading?.trim() || agent.agent_name,
    path: `/marketplace/agent/${params.creator}/${params.slug}`,
    images: agent.agent_image?.slice(0, 1),
    type: "article",
  });
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
    prefetchSimilarAgents(queryClient, params.slug),
  ]);

  const { user } = await getServerUser();
  const { data: creator_agent, status } = await getAgentOrNotFound(
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

async function getAgentOrNotFound(creator: string, slug: string) {
  try {
    return await getV2GetSpecificAgent(creator, slug);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }
}

// The backend limits searches per visitor, so this one carries the visitor's IP.
async function prefetchSimilarAgents(
  queryClient: ReturnType<typeof getQueryClient>,
  slug: string,
) {
  const clientIPToken = await getClientIPToken(await headers());
  return prefetchGetV2ListStoreAgentsQuery(
    queryClient,
    { search_query: slug.replace(/-/g, " ") },
    clientIPToken
      ? {
          request: {
            headers: { [CLIENT_IP_TOKEN_HEADER_NAME]: clientIPToken },
          },
        }
      : undefined,
  );
}
