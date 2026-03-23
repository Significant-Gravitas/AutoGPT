"use client";
import { FeaturedSection } from "../FeaturedSection/FeaturedSection";
import { BecomeACreator } from "../BecomeACreator/BecomeACreator";
import { HeroSection } from "../HeroSection/HeroSection";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { useMainMarketplacePage } from "./useMainMarketplacePage";
import { FeaturedCreators } from "../FeaturedCreators/FeaturedCreators";
import { MainMarketplacePageLoading } from "../MainMarketplacePageLoading";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

function PageSeparator({ className }: { className?: string }) {
  return <hr className={`border-t border-neutral-200 ${className ?? ""}`} />;
}

export function MainMarketplacePage() {
  const { featuredAgents, topAgents, featuredCreators, isLoading, hasError, refetchAll } =
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
              onRetry={refetchAll}
              className="w-full max-w-md"
            />
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-[1360px]">
      <main className="px-4">
        <HeroSection />
        {featuredAgents && (
          <FeaturedSection featuredAgents={featuredAgents.agents} />
        )}
        {/* 100px margin because our featured sections button are placed 40px below the container */}
        <PageSeparator className="mb-6 mt-24" />

        {topAgents && (
          <AgentsSection sectionTitle="Top Agents" agents={topAgents.agents} />
        )}
        <PageSeparator className="mb-[25px] mt-[60px]" />
        {featuredCreators && (
          <FeaturedCreators featuredCreators={featuredCreators.creators} />
        )}
        <PageSeparator className="mb-[25px] mt-[60px]" />
        <BecomeACreator
          title="Become a Creator"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
        />
      </main>
    </div>
  );
}
