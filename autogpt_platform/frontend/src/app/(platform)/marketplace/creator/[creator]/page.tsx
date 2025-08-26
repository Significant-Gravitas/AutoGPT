import { getQueryClient } from "@/lib/react-query/queryClient";
import {
  prefetchGetV2GetCreatorDetailsQuery,
  prefetchGetV2ListStoreAgentsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { MainCreatorPage } from "../../components/MainCreatorPage/MainCreatorPage";

export const dynamic = "force-dynamic";

export interface MarketplaceCreatorPageParams {
  creator: string;
}

// FRONTEND-TODO : Add generateMetadata here

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