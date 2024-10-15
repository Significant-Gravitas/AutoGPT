import * as React from "react";
import { CreatorCard } from "../../CreatorCard";
import { Button } from "@/components/ui/button";
import { useMediaQuery } from "../../../../hooks/useMediaQuery";

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
  const [showAll, setShowAll] = React.useState(false);
  const isMobile = useMediaQuery();

  const toggleShowAll = () => {
    setShowAll(!showAll);
  };

  const displayedCreators =
    isMobile && !showAll ? featuredCreators.slice(0, 2) : featuredCreators;

  return (
    <div className="flex w-full flex-col items-center justify-center gap-6 py-8 lg:items-start">
      <div className="font-neue text-[23px] font-bold leading-9 tracking-tight text-[#282828]">
        Featured creators
      </div>
      <div className="relative w-full">
        <div className="flex flex-wrap items-center justify-center gap-5 lg:justify-start">
          {displayedCreators.map((creator, index) => (
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
          {isMobile && !showAll && featuredCreators.length > 2 && (
            <div className="absolute bottom-0 left-0 right-0 h-1/2 bg-gradient-to-t from-white to-transparent" />
          )}
        </div>
        {isMobile && !showAll && featuredCreators.length > 2 && (
          <div className="relative z-10 w-full text-center">
            <Button onClick={toggleShowAll} variant="link">
              Show more
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};
