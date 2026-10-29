"use client";

import { FilterChips } from "../FilterChips/FilterChips";
import { SearchBar } from "../SearchBar/SearchBar";
import { useHeroSection } from "./useHeroSection";

export const HeroSection = () => {
  const { onFilterChange, searchTerms } = useHeroSection();

  return (
    <div className="mb-16 mt-10 flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-3xl">
        <h1
          // TODO(tokens): SECRT-3152 rare size: md:text-[3rem]; no exact text token: leading-[1.05]
          className="mb-3 text-center text-3xl font-semibold leading-[1.05] tracking-heading text-foreground-strong md:text-[3rem]"
        >
          Hire an AI expert
          <span className="block">
            {/* TODO(tokens): no exact role: text-violet-600 #7C3AED */}
            for <span className="text-violet-600">your team</span>
          </span>
        </h1>
        <p className="mb-8 text-center text-body-md leading-inherit text-foreground-tertiary md:text-lg">
          Ready-made specialists who bring their own skills and workflows
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
