import { GetV2ListStoreAgentsParams } from "@/app/api/__generated__/models/getV2ListStoreAgentsParams";
import { SearchFilterChips } from "@/components/__legacy__/SearchFilterChips";
import { SortDropdown } from "@/components/__legacy__/SortDropdown";
import { Button } from "@/components/atoms/Button/Button";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ArrowLeftIcon } from "@phosphor-icons/react";
import { AgentsSection } from "../../../components/AgentsSection/AgentsSection";
import { FeaturedCreators } from "../../../components/FeaturedCreators/FeaturedCreators";
import { MainSearchResultPageLoading } from "../../../components/MainSearchResultPageLoading";
import { SearchBar } from "../../../components/SearchBar/SearchBar";
import { useMainSearchResultPage } from "./useMainSearchResultPage";

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
      <div className="mx-auto min-h-screen max-w-[1440px] px-10 lg:min-w-[1440px]">
        <div className="mb-4 mt-5">
          <Button
            variant="secondary"
            size="small"
            as="NextLink"
            href="/marketplace"
            leftIcon={<ArrowLeftIcon size={16} />}
          >
            Go back
          </Button>
        </div>
        <div className="flex flex-col gap-4 md:flex-row md:items-center">
          <div className="flex-1">
            <h2 className="text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
              Showing results for:
            </h2>
            <h1 className="font-poppins text-2xl font-semibold leading-[32px] text-neutral-800 dark:text-neutral-100">
              &quot;{searchTerm}&quot;
            </h1>
          </div>
          <div className="flex-none">
            <SearchBar width="w-full md:w-[439px]" height="h-[2.75rem]" />
          </div>
        </div>

        {totalCount > 0 ? (
          <>
            <div className="mt-6 flex flex-col gap-3 md:mt-[36px] md:flex-row md:items-center md:justify-between">
              <SearchFilterChips
                totalCount={totalCount}
                agentsCount={agentsCount}
                creatorsCount={creatorsCount}
                onFilterChange={handleFilterChange}
              />
              <div className="mt-4 md:!mt-0">
                <SortDropdown onSort={handleSortChange} />
              </div>
            </div>
            {/* Content section */}
            <div className="min-h-[500px] max-w-[1440px] space-y-8 py-8">
              {showAgents && agentsCount > 0 && agents && (
                <AgentsSection agents={agents} />
              )}
              <div className="h-[1rem] w-full" />
              {showCreators && creatorsCount > 0 && creators && (
                <FeaturedCreators
                  featuredCreators={creators}
                  title="Creators"
                />
              )}
            </div>
          </>
        ) : (
          <div className="flex min-h-[60vh] flex-col items-center justify-center">
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
