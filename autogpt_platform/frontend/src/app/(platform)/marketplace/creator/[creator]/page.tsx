import {
  getV2GetCreatorDetails,
  prefetchGetV2GetCreatorDetailsQuery,
  prefetchGetV2ListStoreAgentsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { CreatorDetails } from "@/app/api/__generated__/models/creatorDetails";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { Metadata } from "next";
import { notFound } from "next/navigation";
import { MainCreatorPage } from "./components/MainCreatorPage/MainCreatorPage";

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

  let creator: CreatorDetails;
  try {
    const { data } = await getV2GetCreatorDetails(params.creator.toLowerCase());
    creator = data as CreatorDetails;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }

  return {
    title: `${creator.name} - AutoGPT Store`,
    description: creator.description,
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
