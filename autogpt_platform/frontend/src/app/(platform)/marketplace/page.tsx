import {
  prefetchGetV2ListStoreAgentsQuery,
  prefetchGetV2ListStoreCreatorsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { Metadata } from "next";
import { Suspense } from "react";
import { MainMarkeplacePage } from "./components/MainMarketplacePage/MainMarketplacePage";
import { MainMarketplacePageLoading } from "./components/MainMarketplacePageLoading";

export const dynamic = "force-dynamic";

// FIX: Correct metadata
export const metadata: Metadata = {
  title: "Marketplace - AutoGPT Platform",
  description: "Find and use AI Agents created by our community",
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
  openGraph: {
    title: "Marketplace - AutoGPT Platform",
    description: "Find and use AI Agents created by our community",
    type: "website",
    siteName: "AutoGPT Marketplace",
    images: [
      {
        url: "/images/store-og.png",
        width: 1200,
        height: 630,
        alt: "AutoGPT Marketplace",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Marketplace - AutoGPT Platform",
    description: "Find and use AI Agents created by our community",
    images: ["/images/store-twitter.png"],
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
  ]);

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <Suspense fallback={<MainMarketplacePageLoading />}>
        <MainMarkeplacePage />
      </Suspense>
    </HydrationBoundary>
  );
}
