"use client";
import { Separator } from "@/components/__legacy__/ui/separator";
import { FeaturedSection } from "../FeaturedSection/FeaturedSection";
import { BecomeACreator } from "../BecomeACreator/BecomeACreator";
import { HeroSection } from "../HeroSection/HeroSection";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { useMainMarketplacePage } from "./useMainMarketplacePage";
import { FeaturedCreators } from "../FeaturedCreators/FeaturedCreators";
import { MainMarketplacePageLoading } from "../MainMarketplacePageLoading";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

export const MainMarkeplacePage = () => {
  const { featuredAgents, topAgents, featuredCreators, isLoading, hasError } =
    useMainMarketplacePage();

  if (isLoading) {
    return <MainMarketplacePageLoading />;
  }

  if (hasError) {
    return (
      <div className="mx-auto w-screen max-w-[1360px]">
        <main className="px-4">
          <div className="flex min-h-[400px] items-center justify-center">
            <ErrorCard
              isSuccess={false}
              responseError={{ message: "Failed to load marketplace data" }}
              context="marketplace page"
              onRetry={() => window.location.reload()}
              className="w-full max-w-md"
            />
          </div>
        </main>
      </div>
    );
  }

  return (
    // FRONTEND-TODO : Need better state location, need to fetch creators and agents in their respective file, Can't do it right now because these files are used in some other pages of marketplace, will fix it when encounter with those pages
    <div className="mx-auto w-full max-w-[1360px]">
      <main className="px-4">
        <HeroSection />
        {featuredAgents && (
          <FeaturedSection featuredAgents={featuredAgents.agents} />
        )}
        {/* 100px margin because our featured sections button are placed 40px below the container */}
        <Separator className="mb-6 mt-24" />

        {topAgents && (
          <AgentsSection sectionTitle="Top Agents" agents={topAgents.agents} />
        )}
        <Separator className="mb-[25px] mt-[60px]" />
        {featuredCreators && (
          <FeaturedCreators featuredCreators={featuredCreators.creators} />
        )}
        <Separator className="mb-[25px] mt-[60px]" />
        <BecomeACreator
          title="Become a Creator"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
        />
      </main>
    </div>
  );
};
