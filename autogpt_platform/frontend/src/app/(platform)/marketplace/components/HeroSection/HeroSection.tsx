"use client";

import { FilterChips } from "../FilterChips/FilterChips";
import { SearchBar } from "../SearchBar/SearchBar";
import { useHeroSection } from "./useHeroSection";
import { useRouter } from "next/navigation";
import { MessageCircle, Sparkles } from "lucide-react";

export const HeroSection = () => {
  const { onFilterChange } = useHeroSection();
  const router = useRouter();

  const handleDiscoverClick = () => {
    router.push("/marketplace/discover");
  };

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
        
        {/* New AI Discovery CTA */}
        <div className="mb-6 flex justify-center">
          <button
            onClick={handleDiscoverClick}
            className="group relative flex items-center gap-3 rounded-full bg-gradient-to-r from-violet-600 to-purple-600 px-8 py-4 text-white shadow-lg transition-all duration-300 hover:scale-105 hover:shadow-xl"
          >
            <MessageCircle className="h-5 w-5" />
            <span className="text-lg font-medium">Start AI-Powered Discovery</span>
            <Sparkles className="h-5 w-5 animate-pulse" />
            <div className="absolute inset-0 rounded-full bg-white opacity-0 transition-opacity duration-300 group-hover:opacity-10" />
          </button>
        </div>
        
        <div className="mb-3 text-center text-sm text-neutral-600 dark:text-neutral-400">
          or search directly below
        </div>
        
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
