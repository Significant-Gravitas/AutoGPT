import * as React from "react";
import { SearchBar } from "../SearchBar";
import { FilterChips } from "../FilterChips";

interface HeroSectionProps {
  onSearch: (query: string) => void;
  onFilterChange: (selectedFilters: string[]) => void;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  onSearch,
  onFilterChange,
}) => {
  return (
    <div className="mb-2 mt-8 flex flex-col items-center justify-center px-4 sm:mb-4 sm:mt-12 sm:px-6 md:mb-6 md:mt-16 lg:my-24 lg:px-8 xl:my-16">
      <div className="w-full max-w-3xl lg:max-w-4xl xl:max-w-5xl">
        <h1 className="mb-2 text-center font-neue text-3xl font-medium leading-tight tracking-wide text-[#272727] sm:mb-3 sm:text-4xl md:mb-4 md:text-5xl">
          Discover our community made AI Agents
        </h1>
        <p className="mb-4 text-center font-neue text-lg font-medium leading-7 tracking-tight text-[#878787] sm:mb-6 sm:text-xl sm:leading-8 md:mb-8 md:text-2xl md:leading-9 lg:text-[26px]">
          Speed up your workflow with your curated agents
        </p>
        <div className="mb-4 sm:mb-5 md:mb-6">
          <SearchBar onSearch={onSearch} />
        </div>
        <div>
          <div className="flex justify-center">
            <FilterChips
              badges={[
                "Marketing",
                "Sales",
                "Content creation",
                "Lorem ipsum",
                "Lorem ipsum",
              ]}
              onFilterChange={onFilterChange}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
