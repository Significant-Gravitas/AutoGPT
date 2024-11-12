import * as React from "react";
import { HeroSection } from "@/components/agptui/composite/HeroSection";
import {
  FeaturedSection,
  FeaturedAgent,
} from "@/components/agptui/composite/FeaturedSection";
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
import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { Metadata } from "next";

async function getStoreData() {
  const api = new AutoGPTServerAPI();
  const [featuredAgents, topAgents, featuredCreators] = await Promise.all([
    api.getStoreAgents({ featured: true }),
    api.getStoreAgents({ sorted_by: "runs" }),
    api.getStoreCreators({ featured: true, sorted_by: "num_agents" }),
  ]);

  return {
    featuredAgents,
    topAgents,
    featuredCreators,
  };
}

// FIX: Correct metadata
export const metadata: Metadata = {
  title: "Agent Store - NextGen AutoGPT",
  description: "Find and use AI Agents created by our community",
  applicationName: "NextGen AutoGPT Store",
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
    title: "Agent Store - NextGen AutoGPT",
    description: "Find and use AI Agents created by our community",
    type: "website",
    siteName: "NextGen AutoGPT Store",
    images: [
      {
        url: "/images/store-og.png",
        width: 1200,
        height: 630,
        alt: "NextGen AutoGPT Store",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Agent Store - NextGen AutoGPT",
    description: "Find and use AI Agents created by our community",
    images: ["/images/store-twitter.png"],
  },
  icons: {
    icon: "/favicon.ico",
    shortcut: "/favicon-16x16.png",
    apple: "/apple-touch-icon.png",
  },
};

export default async function Page({
  params: { lang },
}: {
  params: { lang: string };
}) {
  // Get data server-side
  const { featuredAgents, topAgents, featuredCreators } = await getStoreData();

  return (
    <div className="mx-auto w-screen max-w-[1360px]">
      <main className="px-4">
        <HeroSection />
        <FeaturedSection
          featuredAgents={featuredAgents.agents as FeaturedAgent[]}
        />
        <Separator />
        <AgentsSection
          sectionTitle="Top Agents"
          agents={topAgents.agents as Agent[]}
        />
        <Separator />
        <FeaturedCreators
          featuredCreators={featuredCreators.creators as FeaturedCreator[]}
        />
        <Separator />
        <BecomeACreator
          title="Want to contribute?"
          heading="We're always looking for more Creators!"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
        />
      </main>
    </div>
  );
}
