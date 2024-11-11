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

// Remove client-side hook since we're doing server-side data fetching
// import { useSupabase } from "@/components/providers/SupabaseProvider";

async function getStoreData() {
  const api = new AutoGPTServerAPI();
  const [featuredAgents, topAgents, featuredCreators] = await Promise.all([
    api.getStoreAgents({ featured: true }),
    api.getStoreAgents({ sorted_by: "runs" }),
    api.getStoreCreators({ featured: true }),
  ]);

  return {
    featuredAgents,
    topAgents,
    featuredCreators,
  };
}

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
