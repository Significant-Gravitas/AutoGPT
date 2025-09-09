import { getQueryClient } from "@/lib/react-query/queryClient";
import {
  getV2GetCreatorDetails,
  prefetchGetV2GetCreatorDetailsQuery,
  prefetchGetV2ListStoreAgentsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { MainCreatorPage } from "../../components/MainCreatorPage/MainCreatorPage";
import { Metadata } from "next";
import { CreatorDetails } from "@/app/api/__generated__/models/creatorDetails";

// Enable ISR with 10-minute revalidation
export const revalidate = 600; // 10 minutes in seconds

export interface MarketplaceCreatorPageParams {
  creator: string;
}

export async function generateMetadata({
  params: _params,
}: {
  params: Promise<MarketplaceCreatorPageParams>;
}): Promise<Metadata> {
  const params = await _params;
  const { data: creator } = await getV2GetCreatorDetails(
    params.creator.toLowerCase(),
  );

  return {
    title: `${(creator as CreatorDetails).name} - AutoGPT Store`,
    description: (creator as CreatorDetails).description,
  };
}

export default async function Page({
  params: _params,
}: {
  params: Promise<MarketplaceCreatorPageParams>;
}) {
  const queryClient = getQueryClient();

  const params = await _params;

  await Promise.all([
    prefetchGetV2ListStoreAgentsQuery(queryClient, {
      creator: params.creator,
    }),
    prefetchGetV2GetCreatorDetailsQuery(queryClient, params.creator),
  ]);

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <MainCreatorPage params={params} />
    </HydrationBoundary>
  );
}
