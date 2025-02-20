"use client";

import * as React from "react";
import { SearchBar } from "@/components/agptui/SearchBar";
import { FilterChips } from "@/components/agptui/FilterChips";
import { useRouter } from "next/navigation";

export const HeroSection: React.FC = () => {
  const router = useRouter();

  function onFilterChange(selectedFilters: string[]) {
    const encodedTerm = encodeURIComponent(selectedFilters.join(", "));
    router.push(`/marketplace/search?searchTerm=${encodedTerm}`);
  }

  return (
    <div className="mb-2 mt-8 flex flex-col items-center justify-center px-4 sm:mb-4 sm:mt-12 sm:px-6 md:mb-6 md:mt-16 lg:my-24 lg:px-8 xl:my-16">
      <div className="w-full max-w-3xl lg:max-w-4xl xl:max-w-5xl">
        <div className="mb-4 text-center md:mb-8">
          <h1 className="text-center">
            <span className="font-poppins text-[48px] font-semibold leading-[54px] text-neutral-950 dark:text-neutral-50">
              Explore AI agents built for{" "}
            </span>
            <span className="font-poppins text-[48px] font-semibold leading-[54px] text-violet-600">
              you
            </span>
            <br />
            <span className="font-poppins text-[48px] font-semibold leading-[54px] text-neutral-950 dark:text-neutral-50">
              by the{" "}
            </span>
            <span className="font-poppins text-[48px] font-semibold leading-[54px] text-blue-500">
              community
            </span>
          </h1>
        </div>
        <h3 className="mb:text-2xl mb-6 text-center font-sans text-xl font-normal leading-loose text-neutral-700 dark:text-neutral-300 md:mb-12">
          Bringing you AI agents designed by thinkers from around the world
        </h3>
        <div className="mb-4 flex justify-center sm:mb-5 md:mb-6">
          <SearchBar height="h-[74px]" />
        </div>
        <div>
          <div className="flex justify-center">
            <FilterChips
              badges={[
                "Marketing",
                "SEO",
                "Content Creation",
                "Automation",
                "Fun",
              ]}
              onFilterChange={onFilterChange}
              multiSelect={false}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
