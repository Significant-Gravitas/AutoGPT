import * as React from "react";
import { CreatorCard } from "@/components/agptui/CreatorCard";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";

interface FeaturedCreator {
  creatorName: string;
  creatorImage: string;
  bio: string;
  agentsUploaded: number;
  avatarSrc: string;
}

interface FeaturedCreatorsProps {
  featuredCreators: FeaturedCreator[];
  onCardClick: (creatorName: string) => void;
}

export const FeaturedCreators: React.FC<FeaturedCreatorsProps> = ({
  featuredCreators,
  onCardClick,
}) => {
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
                  creatorName={creator.creatorName}
                  creatorImage={creator.creatorImage}
                  bio={creator.bio}
                  agentsUploaded={creator.agentsUploaded}
                  avatarSrc={creator.avatarSrc}
                  onClick={() => onCardClick(creator.creatorName)}
                />
              </CarouselItem>
            ))}
          </CarouselContent>
        </Carousel>
        <div className="hidden flex-wrap items-center justify-center gap-3 md:flex lg:justify-start">
          {featuredCreators.map((creator, index) => (
            <CreatorCard
              key={index}
              creatorName={creator.creatorName}
              creatorImage={creator.creatorImage}
              bio={creator.bio}
              agentsUploaded={creator.agentsUploaded}
              avatarSrc={creator.avatarSrc}
              onClick={() => onCardClick(creator.creatorName)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
