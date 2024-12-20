"use client";

import * as React from "react";
import { FeaturedStoreCard } from "@/components/agptui/FeaturedStoreCard";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import { useCallback, useState } from "react";
import { IconLeftArrow, IconRightArrow } from "@/components/ui/icons";
import { useRouter } from "next/navigation";

const BACKGROUND_COLORS = [
  "bg-violet-200 dark:bg-violet-800", // #ddd6fe / #5b21b6
  "bg-blue-200 dark:bg-blue-800", // #bfdbfe / #1e3a8a
  "bg-green-200 dark:bg-green-800", // #bbf7d0 / #065f46
];

export interface FeaturedAgent {
  slug: string;
  agent_name: string;
  agent_image: string;
  creator: string;
  creator_avatar: string;
  sub_heading: string;
  description: string;
  runs: number;
  rating: number;
}

interface FeaturedSectionProps {
  featuredAgents: FeaturedAgent[];
}

export const FeaturedSection: React.FC<FeaturedSectionProps> = ({
  featuredAgents,
}) => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const router = useRouter();

  const handleCardClick = (creator: string, slug: string) => {
    router.push(
      `/store/agent/${encodeURIComponent(creator)}/${encodeURIComponent(slug)}`,
    );
  };

  const handlePrevSlide = useCallback(() => {
    setCurrentSlide((prev) =>
      prev === 0 ? featuredAgents.length - 1 : prev - 1,
    );
  }, [featuredAgents.length]);

  const handleNextSlide = useCallback(() => {
    setCurrentSlide((prev) =>
      prev === featuredAgents.length - 1 ? 0 : prev + 1,
    );
  }, [featuredAgents.length]);

  const getBackgroundColor = (index: number) => {
    return BACKGROUND_COLORS[index % BACKGROUND_COLORS.length];
  };

  return (
    <div className="flex w-full flex-col items-center justify-center">
      <div className="w-full">
        <h2 className="font-poppins mb-8 text-2xl font-semibold leading-7 text-neutral-800 dark:text-neutral-200">
          Featured agents
        </h2>

        <div>
          <Carousel
            opts={{
              loop: true,
              startIndex: currentSlide,
              duration: 500,
              align: "start",
              containScroll: "trimSnaps",
            }}
            className="w-full overflow-x-hidden"
          >
            <CarouselContent className="transition-transform duration-500">
              {featuredAgents.map((agent, index) => (
                <CarouselItem
                  key={index}
                  className="max-w-[460px] flex-[0_0_auto] pr-8"
                >
                  <FeaturedStoreCard
                    agentName={agent.agent_name}
                    subHeading={agent.sub_heading}
                    agentImage={agent.agent_image}
                    creatorName={agent.creator}
                    description={agent.description}
                    runs={agent.runs}
                    rating={agent.rating}
                    backgroundColor={getBackgroundColor(index)}
                    onClick={() => handleCardClick(agent.creator, agent.slug)}
                  />
                </CarouselItem>
              ))}
            </CarouselContent>
          </Carousel>
        </div>

        <div className="mt-8 flex w-full items-center justify-between">
          <div className="flex h-3 items-center gap-2">
            {featuredAgents.map((_, index) => (
              <div
                key={index}
                className={`${
                  currentSlide === index
                    ? "h-3 w-[52px] rounded-[39px] bg-neutral-800 transition-all duration-500 dark:bg-neutral-200"
                    : "h-3 w-3 rounded-full bg-neutral-300 transition-all duration-500 dark:bg-neutral-600"
                }`}
              />
            ))}
          </div>
          <div className="mb-[60px] flex items-center gap-3">
            <button
              onClick={handlePrevSlide}
              className="mb:h-12 mb:w-12 flex h-10 w-10 items-center justify-center rounded-full border border-neutral-400 bg-white dark:border-neutral-600 dark:bg-neutral-800"
            >
              <IconLeftArrow className="h-8 w-8 text-neutral-800 dark:text-neutral-200" />
            </button>
            <button
              onClick={handleNextSlide}
              className="mb:h-12 mb:w-12 flex h-10 w-10 items-center justify-center rounded-full border border-neutral-900 bg-white dark:border-neutral-600 dark:bg-neutral-800"
            >
              <IconRightArrow className="h-8 w-8 text-neutral-800 dark:text-neutral-200" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
