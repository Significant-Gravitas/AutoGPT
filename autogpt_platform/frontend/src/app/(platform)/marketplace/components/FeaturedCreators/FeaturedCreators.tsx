"use client";

import { FadeIn } from "@/components/molecules/FadeIn/FadeIn";
import { StaggeredList } from "@/components/molecules/StaggeredList/StaggeredList";
import { CreatorCard } from "../CreatorCard/CreatorCard";
import { useFeaturedCreators } from "./useFeaturedCreators";
import { Creator } from "@/app/api/__generated__/models/creator";

interface FeaturedCreatorsProps {
  title?: string;
  featuredCreators: Creator[];
}

export const FeaturedCreators = ({
  featuredCreators,
  title = "Featured Creators",
}: FeaturedCreatorsProps) => {
  const { handleCardClick, displayedCreators } = useFeaturedCreators({
    featuredCreators,
  });
  return (
    <div className="flex w-full flex-col items-center justify-center">
      <div className="w-full max-w-[1360px]">
        <FadeIn direction="left" duration={0.5}>
          <h2 className="mb-9 font-poppins text-lg font-semibold text-neutral-800 dark:text-neutral-200">
            {title}
          </h2>
        </FadeIn>

        <StaggeredList
          direction="up"
          staggerDelay={0.1}
          className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4"
        >
          {displayedCreators.map((creator, index) => (
            <CreatorCard
              key={index}
              creatorName={creator.name || creator.username}
              creatorImage={creator.avatar_url}
              bio={creator.description}
              agentsUploaded={creator.num_agents}
              onClick={() => handleCardClick(creator.username)}
              index={index}
            />
          ))}
        </StaggeredList>
      </div>
    </div>
  );
};
