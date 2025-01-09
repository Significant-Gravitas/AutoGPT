"use client";

import * as React from "react";
import { FeaturedStoreCard } from "@/components/agptui/FeaturedStoreCard";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselPrevious,
  CarouselNext,
  CarouselIndicator,
} from "@/components/ui/carousel";
import { useCallback, useState } from "react";
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
      <div className="w-[99vw]">
        <h2 className="font-poppins mx-auto mb-8 max-w-[1360px] px-4 text-2xl font-semibold leading-7 text-neutral-800 dark:text-neutral-200">
          Featured agents
        </h2>

        <div className="w-[99vw] pb-[60px]">
          <Carousel
            className="mx-auto pb-10"
            opts={{
              align: "center",
              containScroll: "trimSnaps",
            }}
          >
            <CarouselContent className="ml-[calc(50vw-690px)]">
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
            <div className="relative mx-auto w-full max-w-[1360px] pl-4">
              <CarouselIndicator />
              <CarouselPrevious afterClick={handlePrevSlide} />
              <CarouselNext afterClick={handleNextSlide} />
            </div>
          </Carousel>
        </div>
      </div>
    </div>
  );
};
