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
import AutoGPTServerAPIServerSide from "@/lib/autogpt-server-api/clientServer";
import { Metadata } from "next";
import { createServerClient } from "@/lib/supabase/server";
import {
  StoreAgentsResponse,
  CreatorsResponse,
} from "@/lib/autogpt-server-api/types";

export const dynamic = "force-dynamic";

async function getStoreData() {
  try {
    const supabase = createServerClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();

    const api = new AutoGPTServerAPIServerSide(
      process.env.NEXT_PUBLIC_AGPT_SERVER_URL,
      process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL,
      supabase,
    );

    // Add error handling and default values
    let featuredAgents: StoreAgentsResponse = {
      agents: [],
      pagination: {
        total_items: 0,
        total_pages: 0,
        current_page: 0,
        page_size: 0,
      },
    };
    let topAgents: StoreAgentsResponse = {
      agents: [],
      pagination: {
        total_items: 0,
        total_pages: 0,
        current_page: 0,
        page_size: 0,
      },
    };
    let featuredCreators: CreatorsResponse = {
      creators: [],
      pagination: {
        total_items: 0,
        total_pages: 0,
        current_page: 0,
        page_size: 0,
      },
    };

    try {
      [featuredAgents, topAgents, featuredCreators] = await Promise.all([
        api.getStoreAgents({ featured: true }),
        api.getStoreAgents({ sorted_by: "runs" }),
        api.getStoreCreators({ featured: true, sorted_by: "num_agents" }),
      ]);
    } catch (error) {
      console.error("Error fetching store data:", error);
    }

    return {
      featuredAgents,
      topAgents,
      featuredCreators,
    };
  } catch (error) {
    console.error("Error in getStoreData:", error);
    return {
      featuredAgents: {
        agents: [],
        pagination: {
          total_items: 0,
          total_pages: 0,
          current_page: 0,
          page_size: 0,
        },
      },
      topAgents: {
        agents: [],
        pagination: {
          total_items: 0,
          total_pages: 0,
          current_page: 0,
          page_size: 0,
        },
      },
      featuredCreators: {
        creators: [],
        pagination: {
          total_items: 0,
          total_pages: 0,
          current_page: 0,
          page_size: 0,
        },
      },
    };
  }
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

export default async function Page({}: {}) {
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
          title="Become a Creator"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
        />
      </main>
    </div>
  );
}
