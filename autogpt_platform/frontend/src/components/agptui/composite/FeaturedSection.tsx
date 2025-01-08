"use client";

import * as React from "react";
import { FeaturedStoreCard } from "@/components/agptui/FeaturedStoreCard";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselPrevious,
  CarouselNext,
} from "@/components/ui/carousel";
import { useCallback, useState } from "react";
import { IconLeftArrow, IconRightArrow } from "@/components/ui/icons";
import { useRouter } from "next/navigation";
import useEmblaCarousel from "embla-carousel-react";

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
  const [carouselRef, api] = useEmblaCarousel();

  const handleCardClick = (creator: string, slug: string) => {
    router.push(
      `/store/agent/${encodeURIComponent(creator)}/${encodeURIComponent(slug)}`,
    );
  };

  const handlePrevSlide = useCallback(() => {
    setCurrentSlide((prev) =>
      prev === 0 ? featuredAgents.length - 1 : prev - 1,
    );
    api?.scrollTo(2);
  }, [featuredAgents.length, api]);

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
      <div className="w-[99vw]">
        <h2 className="font-poppins mx-auto mb-8 max-w-[1360px] px-4 text-2xl font-semibold leading-7 text-neutral-800 dark:text-neutral-200">
          Featured agents
        </h2>

        <div className="w-[99vw] pb-6">
          <Carousel
            className="mx-auto pb-10"
            opts={{
              align: "start",
            }}
          >
            <CarouselContent className="pl-[calc(50vw-680px+16px)] pr-8">
              {featuredAgents.map((agent, index) => (
                <CarouselItem
                  key={index}
                  className="max-w-[460px] flex-[0_0_auto]"
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
            <div className="relative mx-auto w-full max-w-[1360px]">
              <div className="relative top-10 flex h-3 items-center gap-2">
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
              <CarouselPrevious afterClick={handlePrevSlide} />
              <CarouselNext afterClick={handleNextSlide} />
              <button onClick={handlePrevSlide}>Scroll to Slide #3</button>
            </div>
          </Carousel>

          {/* <Carousel
            opts={{
              startIndex: currentSlide,
              duration: 500,
              align: "start",
              containScroll: "trimSnaps",
              slidesToScroll: 1,
            }}
            orientation="horizontal"
            className="w-full overflow-x-hidden"
          >
            <CarouselContent className="pl-[calc((100vw-1360px)/2)]">
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
            <CarouselPrevious />
            <CarouselNext /> */}
          {/* <div className="mx-auto mt-8 flex w-full max-w-[1360px] items-center justify-between">
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
              <div className="mb-[60px] flex items-center gap-3"> */}
          {/* <button
                  onClick={handleNextSlide}
                  className="mb:h-12 mb:w-12 flex h-10 w-10 items-center justify-center rounded-full border border-neutral-900 bg-white dark:border-neutral-600 dark:bg-neutral-800"
                >
                  <IconRightArrow className="h-8 w-8 text-neutral-800 dark:text-neutral-200" />
                </button> */}
          {/* </div> */}
          {/* </div> */}
          {/* </Carousel> */}
        </div>
      </div>
    </div>
  );
};
