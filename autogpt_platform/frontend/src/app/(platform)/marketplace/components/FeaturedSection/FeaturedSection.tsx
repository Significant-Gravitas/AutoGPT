"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import {
  Carousel,
  CarouselContent,
  CarouselIndicator,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/__legacy__/ui/carousel";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { SparkleIcon } from "@phosphor-icons/react";
import Link from "next/link";
import { FeaturedAgentCard } from "../FeaturedAgentCard/FeaturedAgentCard";
import { useFeaturedSection } from "./useFeaturedSection";

const FEATURED_COLORS = [
  "bg-violet-50 border-violet-100/70",
  "bg-blue-50 border-blue-100/70",
  "bg-green-50 border-green-100/70",
];

interface FeaturedSectionProps {
  featuredAgents: StoreAgent[];
}

export const FeaturedSection = ({ featuredAgents }: FeaturedSectionProps) => {
  const { handleNextSlide, handlePrevSlide } = useFeaturedSection({
    featuredAgents,
  });

  return (
    <section className="w-full">
      <div className="mb-8 flex flex-row items-center gap-2">
        <SparkleIcon size={24} />
        <Text variant="h4">Featured Agents</Text>
      </div>

      <Carousel
        opts={{
          align: "center",
          containScroll: "trimSnaps",
        }}
        className="-mx-4"
      >
        <div className="relative">
          <CarouselContent className="px-4">
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
                    backgroundColor={
                      FEATURED_COLORS[index % FEATURED_COLORS.length]
                    }
                  />
                </Link>
              </CarouselItem>
            ))}
          </CarouselContent>
          <div className="pointer-events-none absolute inset-y-0 left-0 w-8 bg-gradient-to-r from-[rgb(246,247,248)] to-transparent" />
          <div className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-[rgb(246,247,248)] to-transparent" />
        </div>
        <div
          className={cn(
            "relative -mt-2",
            featuredAgents.length === 3 && "md:hidden",
          )}
        >
          <CarouselIndicator className="relative top-2 ml-8" />
          <CarouselPrevious
            afterClick={handlePrevSlide}
            className="right-14 h-10 w-10"
          />
          <CarouselNext
            afterClick={handleNextSlide}
            className="right-2 h-10 w-10"
          />
        </div>
      </Carousel>
    </section>
  );
};
