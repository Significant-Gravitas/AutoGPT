import { Metadata } from "next";
import {
  prefetchGetV2ListStoreAgentsQuery,
  prefetchGetV2ListStoreCreatorsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { MainMarkeplacePage } from "./components/MainMarketplacePage/MainMarketplacePage";

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
  icons: {
    icon: "/favicon.ico",
    shortcut: "/favicon-16x16.png",
    apple: "/apple-touch-icon.png",
  },
};

export default async function MarketplacePage(): Promise<React.ReactElement> {
  const queryClient = getQueryClient();

  await Promise.all([
    prefetchGetV2ListStoreAgentsQuery(queryClient, {
      featured: true,
    }),
    prefetchGetV2ListStoreAgentsQuery(queryClient, {
      sorted_by: "runs",
    }),
    prefetchGetV2ListStoreCreatorsQuery(queryClient, {
      featured: true,
      sorted_by: "num_agents",
    }),
  ]);

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <MainMarkeplacePage />
    </HydrationBoundary>
  );
}
