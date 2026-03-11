"use client";

import * as React from "react";
import { SearchBar } from "@/components/__legacy__/SearchBar";
import { FilterChips } from "@/components/__legacy__/FilterChips";
import { useRouter } from "next/navigation";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";

export const HeroSection: React.FC = () => {
  const router = useRouter();
  const onboarding = useOnboarding();

  // Mark marketplace visit task as completed
  React.useEffect(() => {
    onboarding?.completeStep("MARKETPLACE_VISIT");
  }, [onboarding?.completeStep]);

  function onFilterChange(selectedFilters: string[]) {
    const encodedTerm = encodeURIComponent(selectedFilters.join(", "));
    router.push(`/marketplace/search?searchTerm=${encodedTerm}`);
  }

  return (
    <div className="mt-8 mb-2 flex flex-col items-center justify-center px-4 sm:mt-12 sm:mb-4 sm:px-6 md:mt-16 md:mb-6 lg:my-24 lg:px-8 xl:my-16">
      <div className="w-full max-w-3xl lg:max-w-4xl xl:max-w-5xl">
        <div className="mb-4 text-center md:mb-8">
          <h1 className="text-center">
            <span className="font-poppins text-[48px] leading-[54px] font-semibold text-neutral-950 dark:text-neutral-50">
              Explore AI agents built for{" "}
            </span>
            <span className="font-poppins text-[48px] leading-[54px] font-semibold text-violet-600">
              you
            </span>
            <br />
            <span className="font-poppins text-[48px] leading-[54px] font-semibold text-neutral-950 dark:text-neutral-50">
              by the{" "}
            </span>
            <span className="font-poppins text-[48px] leading-[54px] font-semibold text-blue-500">
              community
            </span>
          </h1>
        </div>
        <h3 className="mb:text-2xl mb-6 text-center font-sans text-xl leading-loose font-normal text-neutral-700 md:mb-12 dark:text-neutral-300">
          Bringing you AI agents designed by thinkers from around the world
        </h3>
        <div className="mb-4 flex justify-center sm:mb-5">
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
