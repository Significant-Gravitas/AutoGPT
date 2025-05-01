"use client";

import * as React from "react";
import { SearchBar } from "@/components/agptui/SearchBar";
import { FilterChips } from "@/components/agptui/FilterChips";
import { useRouter } from "next/navigation";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";

export const HeroSection: React.FC = () => {
  const router = useRouter();
  const { completeStep } = useOnboarding();

  // Mark marketplace visit task as completed
  React.useEffect(() => {
    completeStep("MARKETPLACE_VISIT");
  }, [completeStep]);

  function onFilterChange(selectedFilters: string[]) {
    const encodedTerm = encodeURIComponent(selectedFilters.join(", "));
    router.push(`/marketplace/search?searchTerm=${encodedTerm}`);
  }

  return (
    <div className="mx-auto flex w-[90%] flex-col items-center justify-center pb-12 pt-16 md:w-full md:pb-28 md:pt-32">
      {/* Title */}
      <h1 className="mb-4 text-center font-poppins text-2xl font-semibold leading-8 text-zinc-900 md:mb-9 md:text-[2.75rem] md:leading-[3.5rem]">
        <span>Explore AI agents built for </span>
        <span className="text-violet-600">you</span>
        <br />
        <span>by the </span>
        <span className="text-blue-500">community</span>
      </h1>

      {/* Description */}
      <h3 className="mb-6 text-center font-sans text-lg text-zinc-600 md:mb-12 md:text-xl">
        Bringing you AI agents designed by thinkers from around the world
      </h3>

      {/* Seach bar */}
      <SearchBar height="h-14 md:h-18" />

      {/* Filter chips */}
      <div className="mt-5">
        <FilterChips
          badges={["Marketing", "Content Creation", "SEO", "Automation", "Fun"]}
          onFilterChange={onFilterChange}
          multiSelect={false}
        />
      </div>
    </div>
  );
};
