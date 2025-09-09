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
  try {
    const { data: creator } = await getV2GetCreatorDetails(
      params.creator.toLowerCase(),
    );

    return {
      metadataBase: new URL(process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || 'https://platform.agpt.co'),
      title: `${(creator as CreatorDetails).name} - AutoGPT Store`,
      description: (creator as CreatorDetails).description,
    };
  } catch (_error) {
    return {
      metadataBase: new URL(process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || 'https://platform.agpt.co'),
      title: `Creator - AutoGPT Store`,
      description: 'View creator details on AutoGPT Marketplace',
    };
  }
}

export default async function Page({
  params: _params,
}: {
  params: Promise<MarketplaceCreatorPageParams>;
}) {
  const queryClient = getQueryClient();

  const params = await _params;

  try {
    await Promise.all([
      prefetchGetV2ListStoreAgentsQuery(queryClient, {
        creator: params.creator,
      }),
      prefetchGetV2GetCreatorDetailsQuery(queryClient, params.creator),
    ]);
  } catch (error) {
    console.error('Failed to prefetch creator data:', error);
  }

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <MainCreatorPage params={params} />
    </HydrationBoundary>
  );
}
