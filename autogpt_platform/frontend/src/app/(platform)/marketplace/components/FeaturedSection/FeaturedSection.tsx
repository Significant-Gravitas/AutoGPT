"use client";

import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselPrevious,
  CarouselNext,
  CarouselIndicator,
} from "@/components/ui/carousel";
import Link from "next/link";
import { useFeaturedSection } from "./useFeaturedSection";
import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { getBackgroundColor } from "./helper";
import { FeaturedAgentCard } from "../FeaturedAgentCard/FeaturedAgentCard";

interface FeaturedSectionProps {
  featuredAgents: StoreAgent[];
}

export const FeaturedSection = ({ featuredAgents }: FeaturedSectionProps) => {
  const { handleNextSlide, handlePrevSlide } = useFeaturedSection({
    featuredAgents,
  });

  return (
    <section className="w-full">
      <h2 className="mb-8 font-poppins text-2xl font-semibold leading-7 text-neutral-800 dark:text-neutral-200">
        Featured agents
      </h2>

      <Carousel
        opts={{
          align: "center",
          containScroll: "trimSnaps",
        }}
      >
        <CarouselContent>
          {featuredAgents.map((agent, index) => (
            <CarouselItem
              key={index}
              className="h-[480px] md:basis-1/2 lg:basis-1/3"
            >
              <Link
                href={`/marketplace/agent/${encodeURIComponent(agent.creator)}/${encodeURIComponent(agent.slug)}`}
                className="block h-full"
              >
                <FeaturedAgentCard
                  agent={agent}
                  backgroundColor={getBackgroundColor(index)}
                />
              </Link>
            </CarouselItem>
          ))}
        </CarouselContent>
        <div className="relative mt-4">
          <CarouselIndicator />
          <CarouselPrevious afterClick={handlePrevSlide} />
          <CarouselNext afterClick={handleNextSlide} />
        </div>
      </Carousel>
    </section>
  );
};
