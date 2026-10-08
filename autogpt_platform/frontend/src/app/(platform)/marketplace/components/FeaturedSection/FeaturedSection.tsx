"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/molecules/Carousel/Carousel";
import Link from "next/link";
import { FeaturedAgentCard } from "../FeaturedAgentCard/FeaturedAgentCard";
import { SparklesIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { FEATURED_SECTION_ID } from "../MarketplaceTabIntro/helpers";

const FEATURED_COLORS = [
  "bg-purple-50 border-purple-100/70",
  "bg-blue-50 border-blue-100/70",
  "bg-green-50 border-green-100/70",
];

interface FeaturedSectionProps {
  featuredAgents: StoreAgent[];
}

export function FeaturedSection({ featuredAgents }: FeaturedSectionProps) {
  return (
    <section
      id={FEATURED_SECTION_ID}
      className="mb-8 w-full border-b border-zinc-200/70 pb-6"
    >
      <Carousel
        opts={{
          align: "start",
          containScroll: "trimSnaps",
        }}
      >
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-medium tracking-[0.14em] text-purple-600 uppercase">
            <Icon icon={SparklesIcon} size={16} />
            Hand-picked
          </div>
          <div className="flex items-center gap-2">
            <CarouselPrevious className="static h-10 w-10" />
            <CarouselNext className="static h-10 w-10" />
          </div>
        </div>
        <div className="relative -mx-4">
          <CarouselContent className="px-4 pt-1 pb-3">
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
          <div className="pointer-events-none absolute inset-y-0 left-0 w-8 bg-linear-to-r from-background to-transparent" />
          <div className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-linear-to-l from-background to-transparent" />
        </div>
      </Carousel>
    </section>
  );
}
