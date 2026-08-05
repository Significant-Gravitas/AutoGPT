"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/__legacy__/ui/carousel";
import Link from "next/link";
import { FeaturedAgentCard } from "../FeaturedAgentCard/FeaturedAgentCard";
import { SparklesIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

const FEATURED_COLORS = [
  "bg-violet-50 border-violet-100/70",
  "bg-blue-50 border-blue-100/70",
  "bg-green-50 border-green-100/70",
];

interface FeaturedSectionProps {
  featuredAgents: StoreAgent[];
}

export const FeaturedSection = ({ featuredAgents }: FeaturedSectionProps) => {
  return (
    <section className="mb-8 w-full border-b border-zinc-200/70 pb-6">
      <Carousel
        opts={{
          align: "start",
          containScroll: "trimSnaps",
        }}
      >
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-violet-600">
            <Icon icon={SparklesIcon} size={16} />
            Hand-picked
          </div>
          <div className="flex items-center gap-2">
            <CarouselPrevious className="static h-10 w-10" />
            <CarouselNext className="static h-10 w-10" />
          </div>
        </div>
        <div className="relative -mx-4">
          <CarouselContent className="px-4 pb-3 pt-1">
            {featuredAgents.map((agent, index) => (
              <CarouselItem
                key={index}
                className="h-[440px] md:basis-1/2 lg:basis-1/3"
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
      </Carousel>
    </section>
  );
};
