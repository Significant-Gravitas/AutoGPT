"use client";

import * as React from "react";
import { CreatorCard } from "@/components/agptui/CreatorCard";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import { useRouter } from "next/navigation";
export interface FeaturedCreator {
  name: string;
  username: string;
  description: string;
  avatar_url: string;
  num_agents: number;
}

interface FeaturedCreatorsProps {
  featuredCreators: FeaturedCreator[];
}

export const FeaturedCreators: React.FC<FeaturedCreatorsProps> = ({
  featuredCreators,
}) => {
  const router = useRouter();

  const handleCardClick = (creator: string) => {
    router.push(`/store/creator/${creator}`);
  };

  return (
    <div className="flex w-screen flex-col items-center justify-center gap-6 py-8 md:w-full lg:items-start">
      <div className="font-neue text-[23px] font-bold leading-9 tracking-tight text-[#282828]">
        Featured creators
      </div>
      <div className="w-full">
        <Carousel
          className="md:hidden"
          opts={{
            loop: true,
          }}
        >
          <CarouselContent>
            {featuredCreators.map((creator, index) => (
              <CarouselItem
                key={index}
                className="min-w-[13.125rem] max-w-[14.125rem] basis-4/5 sm:basis-3/5"
              >
                <CreatorCard
                  creatorName={creator.username}
                  creatorImage={creator.avatar_url}
                  bio={creator.description}
                  agentsUploaded={creator.num_agents}
                  avatarSrc={creator.avatar_url}
                  onClick={() => handleCardClick(creator.username)}
                />
              </CarouselItem>
            ))}
          </CarouselContent>
        </Carousel>
        <div className="hidden flex-wrap items-center justify-center gap-3 md:flex lg:justify-start">
          {featuredCreators.map((creator, index) => (
            <CreatorCard
              key={index}
              creatorName={creator.username}
              creatorImage={creator.avatar_url}
              bio={creator.description}
              agentsUploaded={creator.num_agents}
              avatarSrc={creator.avatar_url}
              onClick={() => handleCardClick(creator.username)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
