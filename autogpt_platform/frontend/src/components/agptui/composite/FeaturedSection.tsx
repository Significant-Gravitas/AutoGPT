import * as React from "react";
import { FeaturedStoreCard } from "@/components/agptui/FeaturedStoreCard";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import { useCallback, useState } from "react";
import { IconLeftArrow, IconRightArrow } from "@/components/ui/icons";

interface FeaturedAgent {
  agentName: string;
  subHeading: string;
  agentImage: string;
  creatorName: string;
  description: string;
  runs: number;
  rating: number;
}

interface FeaturedSectionProps {
  featuredAgents: FeaturedAgent[];
  onCardClick: (agentName: string) => void;
}

export const FeaturedSection: React.FC<FeaturedSectionProps> = ({
  featuredAgents,
  onCardClick,
}) => {
  const [currentSlide, setCurrentSlide] = useState(0);

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

  return (
    <div className="flex w-full flex-col items-center justify-center px-4 py-8">
      <div className="w-full max-w-[1360px]">
        <h2 className="font-poppins mb-6 text-lg font-semibold leading-7 text-neutral-800">
          Featured agents
        </h2>

        <Carousel
          opts={{
            loop: true,
            startIndex: currentSlide,
            duration: 500,
          }}
          className="w-full"
        >
          <CarouselContent className="gap-5 transition-transform duration-500">
            {" "}
            {/* Add transition */}
            {featuredAgents.map((agent, index) => (
              <CarouselItem key={index} className="basis-full lg:basis-1/3">
                <FeaturedStoreCard
                  agentName={agent.agentName}
                  subHeading={agent.subHeading}
                  agentImage={agent.agentImage}
                  creatorName={agent.creatorName}
                  description={agent.description}
                  runs={agent.runs}
                  rating={agent.rating}
                  onClick={() => onCardClick(agent.agentName)}
                />
              </CarouselItem>
            ))}
          </CarouselContent>
        </Carousel>

        <div className="mt-4 flex w-full items-center justify-between">
          <div className="flex h-3 items-center gap-2">
            {featuredAgents.map((_, index) => (
              <div
                key={index}
                className={`${
                  currentSlide === index
                    ? "h-3 w-[52px] rounded-[39px] bg-neutral-800 transition-all duration-500"
                    : "h-3 w-3 rounded-full bg-neutral-300 transition-all duration-500"
                }`}
              />
            ))}
          </div>
          <div className="flex items-center gap-3">
            {/* We can't get the exact styling of the button using the button from the component library */}
            <button
              onClick={handlePrevSlide}
              className="flex h-[52px] w-[52px] items-center justify-center rounded-full border border-neutral-400 bg-white"
            >
              <IconLeftArrow className="h-8 w-8" />
            </button>
            <button
              onClick={handleNextSlide}
              className="flex h-[52px] w-[52px] items-center justify-center rounded-full border border-neutral-900 bg-white"
            >
              <IconRightArrow className="h-8 w-8" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
