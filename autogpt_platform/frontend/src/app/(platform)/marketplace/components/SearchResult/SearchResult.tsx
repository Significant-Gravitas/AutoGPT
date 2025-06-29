"use client";

import { SearchBar } from "@/app/(platform)/marketplace/components/SearchBar/SearchBar";
import { Separator } from "@/components/ui/separator";
import { SearchFilterChips } from "@/app/(platform)/marketplace/components/SearchFilterChips/SearchFilterChips";
import { SortDropdown } from "@/app/(platform)/marketplace/components/SortDropdown/SortDropdown";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { FeaturedCreators } from "../FeaturedCreators/FeaturedCreators";
import { useSearchResult } from "./useSearchResult";

interface SearchResultsProps {
  searchTerm: string;
  sort: string;
}

export const SearchResults = ({ searchTerm, sort }: SearchResultsProps) => {
  const {
    agents,
    creators,
    isLoading,
    isError,
    agentsCount,
    creatorsCount,
    totalCount,
    showAgents,
    showCreators,
    handleFilterChange,
    handleSortChange,
  } = useSearchResult({ sort, searchTerm });

  // TODO : Add better ui for Error
  if (isError) {
    return "Error..";
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

        {isLoading ? (
          <div className="mt-20 flex flex-col items-center justify-center">
            <p className="text-neutral-500 dark:text-neutral-400">Loading...</p>
          </div>
        ) : totalCount > 0 ? (
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
            <div className="min-h-[500px] max-w-[1440px]">
              {showAgents && agentsCount > 0 && (
                <div className="mt-[36px]">
                  {agents && (
                    <AgentsSection agents={agents} sectionTitle="Agents" />
                  )}
                </div>
              )}

              {showAgents && agentsCount > 0 && creatorsCount > 0 && (
                <Separator />
              )}
              {creators && showCreators && creatorsCount > 0 && (
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
