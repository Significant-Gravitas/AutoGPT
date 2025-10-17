import { SearchBar } from "@/components/__legacy__/SearchBar";
import { useMainSearchResultPage } from "./useMainSearchResultPage";
import { SearchFilterChips } from "@/components/__legacy__/SearchFilterChips";
import { SortDropdown } from "@/components/__legacy__/SortDropdown";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { Separator } from "@/components/__legacy__/ui/separator";
import { FeaturedCreators } from "../FeaturedCreators/FeaturedCreators";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { MainMarketplacePageLoading } from "../MainMarketplacePageLoading";

export const MainSearchResultPage = ({
  searchTerm,
  sort,
}: {
  searchTerm: string;
  sort: string;
}) => {
  const {
    agents,
    creators,
    totalCount,
    agentsCount,
    creatorsCount,
    handleFilterChange,
    handleSortChange,
    showAgents,
    showCreators,
    isAgentsLoading,
    isCreatorsLoading,
    isAgentsError,
    isCreatorsError,
  } = useMainSearchResultPage({ searchTerm, sort });

  const isLoading = isAgentsLoading || isCreatorsLoading;
  const hasError = isAgentsError || isCreatorsError;

  if (isLoading) {
    return <MainMarketplacePageLoading />;
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
      <div className="mx-auto min-h-screen max-w-[1440px] px-10 lg:min-w-[1440px]">
        <div className="mt-8 flex items-center">
          <div className="flex-1">
            <h2 className="text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
              Results for:
            </h2>
            <h1 className="font-poppins text-2xl font-semibold leading-[32px] text-neutral-800 dark:text-neutral-100">
              {searchTerm}
            </h1>
          </div>
          <div className="flex-none">
            <SearchBar width="w-[439px]" height="h-[60px]" />
          </div>
        </div>

        {totalCount > 0 ? (
          <>
            <div className="mt-[36px] flex items-center justify-between">
              <SearchFilterChips
                totalCount={totalCount}
                agentsCount={agentsCount}
                creatorsCount={creatorsCount}
                onFilterChange={handleFilterChange}
              />
              <SortDropdown onSort={handleSortChange} />
            </div>
            {/* Content section */}
            <div className="min-h-[500px] max-w-[1440px] space-y-8 py-8">
              {showAgents && agentsCount > 0 && agents && (
                <div className="mt-[36px]">
                  <AgentsSection agents={agents} sectionTitle="Agents" />
                </div>
              )}

              {showAgents && agentsCount > 0 && creatorsCount > 0 && (
                <Separator />
              )}
              {showCreators && creatorsCount > 0 && creators && (
                <FeaturedCreators
                  featuredCreators={creators}
                  title="Creators"
                />
              )}
            </div>
          </>
        ) : (
          <div className="mt-20 flex flex-col items-center justify-center">
            <h3 className="mb-2 text-xl font-medium text-neutral-600 dark:text-neutral-300">
              No results found
            </h3>
            <p className="text-neutral-500 dark:text-neutral-400">
              Try adjusting your search terms or filters
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
