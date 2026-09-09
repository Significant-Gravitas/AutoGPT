import {
  prefetchGetV2ListMarketplaceSkillsQuery,
  prefetchGetV2ListStoreAgentsQuery,
  prefetchGetV2ListStoreCreatorsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { SHELF_SIZE } from "./components/SkillsSection/helpers";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { buildPageMetadata } from "@/lib/metadata";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { Metadata } from "next";
import { Suspense } from "react";
import { MainMarkeplacePage } from "./components/MainMarketplacePage/MainMarketplacePage";
import { MainMarketplacePageLoading } from "./components/MainMarketplacePageLoading";

export const dynamic = "force-dynamic";

const TITLE = "Marketplace - AutoGPT Platform";
const DESCRIPTION = "Find and use AI Agents created by our community";

// No og:image: the previous /images/store-og.png and store-twitter.png were
// never shipped and 404'd on every unfurl.
export const metadata: Metadata = {
  ...buildPageMetadata({
    title: TITLE,
    description: DESCRIPTION,
    path: "/marketplace",
  }),
  applicationName: "AutoGPT Marketplace",
  authors: [{ name: "AutoGPT Team" }],
  keywords: [
    "AI agents",
    "automation",
    "artificial intelligence",
    "AutoGPT",
    "marketplace",
  ],
  robots: {
    index: true,
    follow: true,
  },
};

export default async function MarketplacePage(): Promise<React.ReactElement> {
  const queryClient = getQueryClient();

  // Prefetch all data on server with proper caching
  await Promise.all([
    prefetchGetV2ListStoreAgentsQuery(
      queryClient,
      { featured: true },
      {
        query: {
          staleTime: 60 * 1000, // 60 seconds
          gcTime: 5 * 60 * 1000, // 5 minutes (formerly cacheTime)
        },
      },
    ),
    prefetchGetV2ListStoreAgentsQuery(
      queryClient,
      { sorted_by: "runs", page_size: 1000 },
      {
        query: {
          staleTime: 60 * 1000, // 60 seconds
          gcTime: 5 * 60 * 1000, // 5 minutes
        },
      },
    ),
    prefetchGetV2ListStoreCreatorsQuery(
      queryClient,
      { featured: true, sorted_by: "num_agents" },
      {
        query: {
          staleTime: 60 * 1000, // 60 seconds
          gcTime: 5 * 60 * 1000, // 5 minutes
        },
      },
    ),
    // With the flag off the endpoint 404s; prefetch swallows that and the
    // client gate hides the shelf either way.
    prefetchGetV2ListMarketplaceSkillsQuery(
      queryClient,
      { page_size: SHELF_SIZE },
      {
        query: {
          staleTime: 60 * 1000, // 60 seconds
          gcTime: 5 * 60 * 1000, // 5 minutes
        },
      },
    ),
  ]);

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <Suspense fallback={<MainMarketplacePageLoading />}>
        <MainMarkeplacePage />
      </Suspense>
    </HydrationBoundary>
  );
}
