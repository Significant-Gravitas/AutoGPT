import {
  getV2GetCreatorDetails,
  prefetchGetV2GetCreatorDetailsQuery,
  prefetchGetV2ListStoreAgentsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { CreatorDetails } from "@/app/api/__generated__/models/creatorDetails";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { Metadata } from "next";
import { MainCreatorPage } from "../../components/MainCreatorPage/MainCreatorPage";

export const dynamic = "force-dynamic";

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
