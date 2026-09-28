"use client";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { UserAiIcon } from "@hugeicons/core-free-icons";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import {
  Flag,
  useFlagStatus,
  useGetFlag,
} from "@/services/feature-flags/use-get-flag";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { CategoryFilter } from "../CategoryFilter/CategoryFilter";
import { BecomeACreator } from "../BecomeACreator/BecomeACreator";
import { FeaturedSection } from "../FeaturedSection/FeaturedSection";
import { ExpertsSection } from "../ExpertsSection/ExpertsSection";
import { SkillsList } from "../SkillsList/SkillsList";
import { SkillsSection } from "../SkillsSection/SkillsSection";
import { WorkflowsShelf } from "../WorkflowsShelf/WorkflowsShelf";
import { HeroSection } from "../HeroSection/HeroSection";
import { MainMarketplacePageLoading } from "../MainMarketplacePageLoading";
import { MarketplaceTabIntro } from "../MarketplaceTabIntro/MarketplaceTabIntro";
import { AGENTS_SECTION_ID } from "../MarketplaceTabIntro/helpers";
import { useMainMarketplacePage } from "./useMainMarketplacePage";

export const MainMarkeplacePage = () => {
  const {
    featuredAgents,
    topAgents,
    category,
    setCategory,
    isLoading,
    hasError,
  } = useMainMarketplacePage();
  const { isLoggedIn, isUserLoading } = useAuth();
  const isHireExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  // Branch on `ready` so the shelf does not pop in after LaunchDarkly
  // answers and push the whole workflow catalogue down.
  const skillsHub = useFlagStatus(Flag.SKILLS_HUB);
  // Hiring is still behind the flag, but the expert pages are public: a
  // visitor browsing the marketplace needs a way to reach them. Signed-in
  // users keep the flag gate so the beta stays invisible to them.
  const showExperts = !isUserLoading && (!isLoggedIn || isHireExpertsEnabled);
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
        {isHireExpertsEnabled ? (
          <>
            {skillsHub.ready && skillsHub.enabled ? (
              <SkillsList key={category ?? "all"} category={category} />
            ) : null}
            {topAgents && (
              <WorkflowsShelf
                id={AGENTS_SECTION_ID}
                agents={topAgents.agents}
                featuredAgents={category ? [] : (featuredAgents?.agents ?? [])}
                total={topAgents.pagination.total_items}
              />
            )}
          </>
        ) : (
          <>
            {skillsHub.ready && skillsHub.enabled ? (
              <SkillsSection category={category} />
            ) : null}
            {topAgents && (
              <div className="mb-20" id={AGENTS_SECTION_ID}>
                <AgentsSection
                  sectionTitle="All AI Workflows"
                  titleIcon={
                    <Icon icon={UserAiIcon} size="2.2rem" aria-hidden />
                  }
                  subtitle="Ready-made automations from the community."
                  agents={topAgents.agents}
                >
                  {/* Featured is a whole-marketplace shelf; under a category filter
                      it would show workflows the filter excludes. */}
                  {!category &&
                    featuredAgents &&
                    featuredAgents.agents.length > 0 && (
                      <FeaturedSection featuredAgents={featuredAgents.agents} />
                    )}
                </AgentsSection>
              </div>
            )}
          </>
        )}
        {/* The expert layout invites publishing from under the workflows
            shelf instead, where it cannot outshout the shelf itself. */}
        {!isHireExpertsEnabled || !hasWorkflowShelf ? (
          <BecomeACreator
            title="Become a Creator"
            description="Join our ever-growing community of hackers and tinkerers"
            buttonText="Become a Creator"
          />
        ) : null}
      </main>
      <MarketplaceTabIntro />
    </div>
  );
};
