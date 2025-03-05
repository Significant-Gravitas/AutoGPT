"use client";

import * as React from "react";
import { FeaturedAgentCard } from "@/components/agptui/FeaturedAgentCard";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselPrevious,
  CarouselNext,
  CarouselIndicator,
} from "@/components/ui/carousel";
import { useCallback, useState } from "react";
import { StoreAgent } from "@/lib/autogpt-server-api";
import Link from "next/link";

const BACKGROUND_COLORS = [
  "bg-violet-200 dark:bg-violet-800", // #ddd6fe / #5b21b6
  "bg-blue-200 dark:bg-blue-800", // #bfdbfe / #1e3a8a
  "bg-green-200 dark:bg-green-800", // #bbf7d0 / #065f46
];

interface FeaturedSectionProps {
  featuredAgents: StoreAgent[];
}

export const FeaturedSection: React.FC<FeaturedSectionProps> = ({
  featuredAgents,
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

  const getBackgroundColor = (index: number) => {
    return BACKGROUND_COLORS[index % BACKGROUND_COLORS.length];
  };

  return (
    <section className="mx-auto w-full max-w-7xl px-4 pb-16">
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
