"use client";

import { CreatorDetails } from "@/app/api/__generated__/models/creatorDetails";
import { CreatorCard } from "../CreatorCard/CreatorCard";
import { SectionHeader } from "../SectionHeader";
import { useFeaturedCreators } from "./useFeaturedCreators";
import { UserCircle02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
        <SectionHeader
          eyebrow="The community"
          eyebrowIcon={<Icon icon={UserCircle02Icon} size={16} />}
          title={title}
          subtitle="The people behind the workflows you install."
        />

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
