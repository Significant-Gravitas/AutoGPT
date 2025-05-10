"use client";

import * as React from "react";
import { CreatorCard } from "@/components/agptui/CreatorCard";
import { useRouter } from "next/navigation";

export interface FeaturedCreator {
  name: string;
  username: string;
  description: string;
  avatar_url: string;
  num_agents: number;
}

interface FeaturedCreatorsProps {
  title?: string;
  featuredCreators: FeaturedCreator[];
}

export const FeaturedCreators: React.FC<FeaturedCreatorsProps> = ({
  featuredCreators,
  title = "Featured Creators",
}) => {
  const router = useRouter();

  const handleCardClick = (creator: string) => {
    router.push(`/marketplace/creator/${encodeURIComponent(creator)}`);
  };

  // Only show first 4 creators
  const displayedCreators = featuredCreators.slice(0, 4);

  return (
    <div className="w-full space-y-9">
      <h2 className="font-poppins text-base font-medium text-zinc-500 dark:text-zinc-200">
        {title}
      </h2>

      <div className="flex flex-wrap gap-5">
        {displayedCreators.map((creator, index) => (
          <CreatorCard
            key={index}
            creatorName={creator.name || creator.username}
            creatorImage={creator.avatar_url}
            bio={creator.description}
            agentsUploaded={creator.num_agents}
            onClick={() => handleCardClick(creator.username)}
          />
        ))}
      </div>
    </div>
  );
};
