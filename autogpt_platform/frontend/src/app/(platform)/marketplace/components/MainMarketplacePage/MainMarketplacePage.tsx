"use client";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";
import type { PublishState } from "@/components/contextual/PublishAgentModal/usePublishAgentModal";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useState } from "react";
import { BecomeACreator } from "../BecomeACreator/BecomeACreator";
import { CategoryFilter } from "../CategoryFilter/CategoryFilter";
import { ExpertsSection } from "../ExpertsSection/ExpertsSection";
import { HeroSection } from "../HeroSection/HeroSection";
import { MainMarketplacePageLoading } from "../MainMarketplacePageLoading";
import { MarketplaceTabIntro } from "../MarketplaceTabIntro/MarketplaceTabIntro";
import { AGENTS_SECTION_ID } from "../MarketplaceTabIntro/helpers";
import { SkillsList } from "../SkillsList/SkillsList";
import { WorkflowsShelf } from "../WorkflowsShelf/WorkflowsShelf";
import { useMainMarketplacePage } from "./useMainMarketplacePage";

export function MainMarkeplacePage() {
  const [publishState, setPublishState] = useState<PublishState>({
    isOpen: false,
    step: "select",
    submissionData: null,
  });
  function openPublishing() {
    setPublishState({ isOpen: true, step: "select", submissionData: null });
  }
  const {
    featuredAgents,
    topAgents,
    category,
    setCategory,
    isLoading,
    hasError,
  } = useMainMarketplacePage();
  const { isUserLoading } = useAuth();

  const showExperts = !isUserLoading;
  const hasWorkflowShelf = Boolean(
    topAgents &&
      (topAgents.agents.length > 0 ||
        (!category && (featuredAgents?.agents.length ?? 0) > 0)),
  );

  if (isLoading) {
    return <MainMarketplacePageLoading />;
  }

  if (hasError) {
    return (
      <div className="mx-auto w-full max-w-[1360px]">
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
        {/* Above all three shelves because it narrows all three: a filter
            below its content changes what the reader has scrolled past. */}
        <CategoryFilter selected={category} onSelect={setCategory} />
        {showExperts ? <ExpertsSection category={category} /> : null}

        <SkillsList key={category ?? "all"} category={category} />
        {topAgents && (
          <WorkflowsShelf
            id={AGENTS_SECTION_ID}
            agents={topAgents.agents}
            featuredAgents={category ? [] : (featuredAgents?.agents ?? [])}
            total={topAgents.pagination.total_items}
            onPublish={openPublishing}
          />
        )}

        {/* The expert layout invites publishing from under the workflows
            shelf instead, where it cannot outshout the shelf itself. */}
        {!hasWorkflowShelf ? (
          <BecomeACreator
            description="Join our ever-growing community of hackers and tinkerers"
            buttonText="Publish your workflows"
            onPublish={openPublishing}
          />
        ) : null}
      </main>
      <PublishAgentModal
        showTrigger={false}
        targetState={publishState}
        onStateChange={setPublishState}
      />
      <MarketplaceTabIntro canShow={!publishState.isOpen} />
    </div>
  );
}
