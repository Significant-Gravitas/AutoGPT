import { GetV2ListStoreAgentsParams } from "@/app/api/__generated__/models/getV2ListStoreAgentsParams";
import { SearchFilterChips } from "@/components/__legacy__/SearchFilterChips";
import { SortDropdown } from "@/components/__legacy__/SortDropdown";
import { Button } from "@/components/atoms/Button/Button";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { AgentsSection } from "../../../components/AgentsSection/AgentsSection";
import { FeaturedCreators } from "../../../components/FeaturedCreators/FeaturedCreators";
import { MainSearchResultPageLoading } from "../../../components/MainSearchResultPageLoading";
import { SearchBar } from "../../../components/SearchBar/SearchBar";
import { SectionHeader } from "../../../components/SectionHeader";
import { SkillCard } from "../../../components/SkillsSection/components/SkillCard";
import { ExpertCard } from "../../../components/ExpertsSection/components/ExpertCard";
import { useMainSearchResultPage } from "./useMainSearchResultPage";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

type MarketplaceSearchSort = GetV2ListStoreAgentsParams["sorted_by"];

export const MainSearchResultPage = ({
  searchTerm,
  sort,
}: {
  searchTerm: string;
  sort: MarketplaceSearchSort;
}) => {
  const {
    agents,
    creators,
    skills,
    experts,
    hiredTemplateIds,
    totalCount,
    agentsCount,
    creatorsCount,
    skillsCount,
    expertsCount,
    handleFilterChange,
    handleSortChange,
    showAgents,
    showCreators,
    showSkills,
    showExperts,
    isExpertsVisible,
    isSkillsHubEnabled,
    isAgentsLoading,
    isCreatorsLoading,
    isExpertsLoading,
    isAgentsError,
    isCreatorsError,
  } = useMainSearchResultPage({ searchTerm, sort });

  // Experts gate the skeleton too: without it a roster still in flight reads
  // as zero results, and a search matching only an expert says "No results
  // found" before the card arrives.
  const isLoading = isAgentsLoading || isCreatorsLoading || isExpertsLoading;
  const hasError = isAgentsError || isCreatorsError;

  if (isLoading) {
    return <MainSearchResultPageLoading />;
  }

  if (hasError) {
    return (
      <div className="flex min-h-[500px] items-center justify-center">
        <ErrorCard
          isSuccess={false}
          responseError={{ message: "Failed to load marketplace data" }}
          context="marketplace page"
          onRetry={() => window.location.reload()}
        />
      </div>
    );
  }
  return (
    <div className="w-full">
      <div className="mx-auto min-h-screen w-full max-w-[1440px] px-6 md:px-10">
        <div className="mb-4 mt-5">
          <Button
            variant="secondary"
            size="small"
            as="NextLink"
            href="/marketplace"
            leftIcon={<Icon icon={ArrowLeft02Icon} size={16} />}
          >
            Go back
          </Button>
        </div>
        <div className="flex flex-col gap-4 md:flex-row md:items-center">
          <div className="flex-1">
            <Text
              variant="large-medium"
              as="h2"
              className="leading-normal text-zinc-800"
            >
              Showing results for:
            </Text>
            <Text
              variant="h4"
              as="h1"
              unmask={false}
              className="text-2xl font-semibold leading-8 text-zinc-800"
            >
              &quot;{searchTerm}&quot;
            </Text>
          </div>
          <div className="flex-none">
            <SearchBar width="w-full md:w-[439px]" height="h-11" />
          </div>
        </div>

        {totalCount > 0 ? (
          <>
            <div className="mt-6 flex flex-col gap-3 md:mt-9 md:flex-row md:items-center md:justify-between">
              <SearchFilterChips
                totalCount={totalCount}
                agentsCount={agentsCount}
                creatorsCount={creatorsCount}
                skillsCount={isSkillsHubEnabled ? skillsCount : undefined}
                expertsCount={isExpertsVisible ? expertsCount : undefined}
                onFilterChange={handleFilterChange}
              />
              <div className="mt-4 md:mt-0">
                <SortDropdown onSort={handleSortChange} />
              </div>
            </div>
            {/* Content section */}
            <div className="min-h-[500px] max-w-[1440px] space-y-8 py-8">
              {showExperts && expertsCount > 0 ? (
                <section aria-labelledby="search-experts-heading">
                  <SectionHeader
                    title="Experts"
                    titleId="search-experts-heading"
                  />
                  <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-3">
                    {experts.map((expert) => (
                      <ExpertCard
                        key={expert.id}
                        expert={expert}
                        isHired={hiredTemplateIds.has(expert.id)}
                      />
                    ))}
                  </div>
                </section>
              ) : null}
              {showAgents && agentsCount > 0 && agents && (
                <AgentsSection agents={agents} />
              )}
              {showSkills && skillsCount > 0 ? (
                <section aria-labelledby="search-skills-heading">
                  <SectionHeader
                    title="Skills"
                    titleId="search-skills-heading"
                  />
                  <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                    {skills.map((skill) => (
                      <SkillCard
                        key={skill.slug}
                        skill={skill}
                        isInstalled={false}
                      />
                    ))}
                  </div>
                </section>
              ) : null}
              <div className="h-4 w-full" />
              {showCreators && creatorsCount > 0 && creators && (
                <FeaturedCreators featuredCreators={creators} />
              )}
            </div>
          </>
        ) : (
          <div className="flex min-h-[60vh] flex-col items-center justify-center">
            <Text
              variant="lead-medium"
              as="h3"
              tone="secondary"
              className="mb-2"
            >
              No results found
            </Text>
            <Text variant="large" tone="muted">
              Try adjusting your search terms or filters
            </Text>
          </div>
        )}
      </div>
    </div>
  );
};
