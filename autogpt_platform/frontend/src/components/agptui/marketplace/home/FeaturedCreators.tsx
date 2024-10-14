import * as React from "react";
import { CreatorCard } from "../../CreatorCard";

interface FeaturedCreator {
  creatorName: string;
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
    <div className="flex w-full flex-col items-start justify-center gap-6 py-8">
      <div className="font-neue text-[23px] font-bold leading-9 tracking-tight text-[#282828]">
        Featured creators
      </div>
      <div className="flex flex-wrap items-center justify-start gap-5">
        {featuredCreators.map((creator, index) => (
          <CreatorCard
            key={index}
            creatorName={creator.creatorName}
            bio={creator.bio}
            agentsUploaded={creator.agentsUploaded}
            avatarSrc={creator.avatarSrc}
            onClick={() => onCardClick(creator.creatorName)}
          />
        ))}
      </div>
    </div>
  );
};
