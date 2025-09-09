import { Metadata } from "next";
import {
  prefetchGetV2ListStoreAgentsQuery,
  prefetchGetV2ListStoreCreatorsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { MainMarkeplacePage } from "./components/MainMarketplacePage/MainMarketplacePage";

// Enable ISR with 10-minute revalidation
export const revalidate = 600; // 10 minutes in seconds

// FIX: Correct metadata
export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || 'https://platform.agpt.co'),
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

  // Try to prefetch data but don't fail if the API is down
  // The client-side will handle fetching with proper error handling
  try {
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
  } catch (error) {
    // Log the error but don't fail the page render
    console.error('Failed to prefetch marketplace data:', error);
  }

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <MainMarkeplacePage />
    </HydrationBoundary>
  );
}
