"use client";

import { CreatorDetails } from "@/app/api/__generated__/models/creatorDetails";
import { Text } from "@/components/atoms/Text/Text";
import { UserCircleDashedIcon } from "@phosphor-icons/react";
import { CreatorCard } from "../CreatorCard/CreatorCard";
import { useFeaturedCreators } from "./useFeaturedCreators";

interface FeaturedCreatorsProps {
  title?: string;
  featuredCreators: CreatorDetails[];
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
        <div className="mb-8 flex flex-row items-center gap-2">
          <UserCircleDashedIcon size={24} />
          <Text variant="h4">{title}</Text>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
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
        </div>
      </div>
    </div>
  );
};
