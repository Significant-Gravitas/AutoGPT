import * as React from "react";
import { SearchBar } from "@/components/agptui/SearchBar";
import { FilterChips } from "@/components/agptui/FilterChips";

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
        <div className="8md:mb-8 mb-4 text-center">
          <span className="font-['Poppins'] text-4xl font-semibold leading-[54px] text-neutral-950 md:text-5xl">
            Explore AI agents built for{" "}
          </span>
          <span className="font-['Poppins'] text-4xl font-semibold leading-[54px] text-violet-600 md:text-5xl">
            you
            <br />
          </span>
          <span className="font-['Poppins'] text-4xl font-semibold leading-[54px] text-neutral-950 md:text-5xl">
            by the{" "}
          </span>
          <span className="font-['Poppins'] text-4xl font-semibold leading-[54px] text-blue-500 md:text-5xl">
            community
          </span>
        </div>
        <div className="mb:text-2xl mb-6 text-center font-['Geist'] text-xl font-normal leading-loose text-neutral-700 md:mb-12">
          Bringing you AI agents designed by thinkers from around the world
        </div>
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
