"use client";

import { cn } from "@/lib/utils";
import { FilterChips } from "../FilterChips/FilterChips";
import { SearchBar } from "../SearchBar/SearchBar";
import { useHeroSection } from "./useHeroSection";

const textStyles =
  "font-poppins text-[2.45rem] leading-[2.75rem] md:text-[3rem] font-semibold md:leading-[3.375rem]";

export const HeroSection = () => {
  const { onFilterChange, searchTerms } = useHeroSection();
  return (
    <div className="mb-12 mt-8 flex flex-col items-center justify-center px-4 sm:mt-12 sm:px-6 md:mt-16 lg:my-24 lg:px-8 xl:my-16">
      <div className="w-full max-w-3xl lg:max-w-4xl xl:max-w-5xl">
        <div className="mb-4 text-center md:mb-8">
          <h1 className="text-center">
            <span
              className={cn(
                textStyles,
                "text-neutral-950 dark:text-neutral-50",
              )}
            >
              Explore AI agents built for{" "}
            </span>
            <span className={cn(textStyles, "text-violet-600")}>you</span>
            <br />
            <span
              className={cn(
                textStyles,
                "text-neutral-950 dark:text-neutral-50",
              )}
            >
              by the{" "}
            </span>
            <span className={cn(textStyles, "text-blue-500")}>community</span>
          </h1>
        </div>
        <h3 className="mb-6 text-center font-sans text-lg font-normal leading-normal text-neutral-700 dark:text-neutral-300 md:mb-12 md:text-xl">
          Bringing you AI agents designed by thinkers from around the world
        </h3>
        <div className="mb-4 flex w-full justify-center sm:mb-5">
          <SearchBar />
        </div>
        <div>
          <div className="flex justify-center">
            <FilterChips
              badges={searchTerms}
              onFilterChange={onFilterChange}
              multiSelect={false}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
