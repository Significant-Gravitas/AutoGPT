import React from "react";
import { HeroSection } from "@/components/agptui/composite/HeroSection";
import { FeaturedSection } from "@/components/agptui/composite/FeaturedSection";
import {
  AgentsSection,
  Agent,
} from "@/components/agptui/composite/AgentsSection";
import { BecomeACreator } from "@/components/agptui/BecomeACreator";
import {
  FeaturedCreators,
  FeaturedCreator,
} from "@/components/agptui/composite/FeaturedCreators";
import { Separator } from "@/components/ui/separator";
import { Metadata } from "next";

import { getMarketplaceData } from "./actions";

// Force dynamic rendering to avoid static generation issues with cookies
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
  const { featuredAgents, topAgents, featuredCreators } =
    await getMarketplaceData();

  return (
    <div className="mx-auto w-screen max-w-[1360px]">
      <main className="px-4">
        <HeroSection />
        <FeaturedSection featuredAgents={featuredAgents.agents} />
        {/* 100px margin because our featured sections button are placed 40px below the container */}
        <Separator className="mb-6 mt-24" />
        <AgentsSection
          sectionTitle="Top Agents"
          agents={topAgents.agents as Agent[]}
        />
        <Separator className="mb-[25px] mt-[60px]" />
        <FeaturedCreators
          featuredCreators={featuredCreators.creators as FeaturedCreator[]}
        />
        <Separator className="mb-[25px] mt-[60px]" />
        <BecomeACreator
          title="Become a Creator"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
        />
      </main>
    </div>
  );
}
