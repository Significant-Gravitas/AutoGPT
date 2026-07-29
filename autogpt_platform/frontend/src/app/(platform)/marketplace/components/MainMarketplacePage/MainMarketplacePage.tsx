"use client";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { AICatalogIcon } from "../AICatalogIcon";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { BecomeACreator } from "../BecomeACreator/BecomeACreator";
import { FeaturedCreators } from "../FeaturedCreators/FeaturedCreators";
import { FeaturedSection } from "../FeaturedSection/FeaturedSection";
import { ExpertsSection } from "../ExpertsSection/ExpertsSection";
import { HeroSection } from "../HeroSection/HeroSection";
import { MainMarketplacePageLoading } from "../MainMarketplacePageLoading";
import { useMainMarketplacePage } from "./useMainMarketplacePage";

export const MainMarkeplacePage = () => {
  const { featuredAgents, topAgents, featuredCreators, isLoading, hasError } =
    useMainMarketplacePage();
  const isHireExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);

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
      <main className="px-6 pb-16 md:px-10 lg:px-14">
        <HeroSection />
        {isHireExpertsEnabled ? <ExpertsSection /> : null}
        {topAgents && (
          <div className="mb-20">
            <AgentsSection
              sectionTitle="All AI Workflows"
              titleIcon={<AICatalogIcon size={30} />}
              subtitle={
                isHireExpertsEnabled
                  ? "Install one on an Expert, or run it standalone."
                  : "Ready-made automations from the community."
              }
              agents={topAgents.agents}
            >
              {featuredAgents && featuredAgents.agents.length > 0 && (
                <FeaturedSection featuredAgents={featuredAgents.agents} />
              )}
            </AgentsSection>
          </div>
        )}
        {featuredCreators && (
          <div className="mb-4">
            <FeaturedCreators featuredCreators={featuredCreators.creators} />
          </div>
        )}
        <BecomeACreator
          title="Become a Creator"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
        />
      </main>
    </div>
  );
};
