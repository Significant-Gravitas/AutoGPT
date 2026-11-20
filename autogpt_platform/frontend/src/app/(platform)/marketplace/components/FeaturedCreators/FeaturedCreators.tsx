"use client";

import { CreatorDetails } from "@/app/api/__generated__/models/creatorDetails";
import { CreatorCard } from "../CreatorCard/CreatorCard";
import { useFeaturedCreators } from "./useFeaturedCreators";

interface Props {
  featuredCreators: CreatorDetails[];
}

export function FeaturedCreators({ featuredCreators }: Props) {
  const { handleCardClick, displayedCreators } = useFeaturedCreators({
    featuredCreators,
  });
  return (
    <div className="flex w-full flex-col items-center justify-center">
      <div className="w-full max-w-[1360px]">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
          {displayedCreators.map((creator, index) => (
            <CreatorCard
              key={index}
              creatorName={creator.name || creator.username}
              creatorUsername={creator.username}
              creatorImage={creator.avatar_url}
              bio={creator.description}
              agentsUploaded={creator.num_agents}
              onClick={() => handleCardClick(creator.username)}
              index={index}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
