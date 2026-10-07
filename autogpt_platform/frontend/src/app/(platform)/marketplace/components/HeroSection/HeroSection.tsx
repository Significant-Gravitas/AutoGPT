"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { FilterChips } from "../FilterChips/FilterChips";
import { SearchBar } from "../SearchBar/SearchBar";
import { useHeroSection } from "./useHeroSection";

export function HeroSection() {
  const { onFilterChange, searchTerms } = useHeroSection();
  const isHireExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);

  return (
    <div className="mt-10 mb-16 flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-3xl">
        <Text
          variant="lead-semibold"
          as="h1"
          tone="primary"
          className="mb-3 text-center text-3xl leading-[1.05] tracking-[-0.02em] md:text-[3rem]"
        >
          {isHireExpertsEnabled ? (
            <>
              Hire an AI expert
              <span className="block">
                for <span className="text-purple-600">your team</span>
              </span>
            </>
          ) : (
            <>
              Explore AI agents built for{" "}
              <span className="text-purple-600">you</span>
              <span className="block">by the community</span>
            </>
          )}
        </Text>
        <Text
          variant="large"
          tone="muted"
          className="mb-8 text-center text-[15px] md:text-lg"
        >
          {isHireExpertsEnabled ? (
            <>
              Ready-made specialists who bring their own skills and workflows
              <span className="block">— working in minutes.</span>
            </>
          ) : (
            "Bringing you AI agents designed by thinkers from around the world"
          )}
        </Text>
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
}
