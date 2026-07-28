"use client";

import { FilterChips } from "../FilterChips/FilterChips";
import { SearchBar } from "../SearchBar/SearchBar";
import { useHeroSection } from "./useHeroSection";

export const HeroSection = () => {
  const { onFilterChange, searchTerms } = useHeroSection();
  return (
    <div className="mb-16 mt-10 flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-3xl">
        <h1 className="mb-3 text-center text-3xl font-semibold tracking-[-0.02em] text-zinc-900 md:text-4xl">
          AI experts and workflows,
          <span className="block">
            built for <span className="text-violet-600">your team</span>
          </span>
        </h1>
        <p className="mb-8 text-center text-[15px] text-zinc-500 md:text-lg">
          Hire a ready-made specialist, or install automations from the
          community
          <span className="block">— working in minutes.</span>
        </p>
        <div className="mb-4 flex w-full justify-center">
          <SearchBar />
        </div>
        <div className="flex justify-center">
          <FilterChips
            badges={searchTerms}
            onFilterChange={onFilterChange}
            multiSelect={false}
          />
        </div>
      </div>
    </div>
  );
};
