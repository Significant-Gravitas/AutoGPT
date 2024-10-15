import * as React from "react";
import { StarIcon, StarFilledIcon } from "@radix-ui/react-icons";
import Image from "next/image";

interface FeaturedStoreCardProps {
  agentName: string;
  agentImage: string;
  creatorName: string;
  description: string;
  runs: number;
  rating: number;
  onClick: () => void;
}

export const FeaturedStoreCard: React.FC<FeaturedStoreCardProps> = ({
  agentName,
  agentImage,
  creatorName,
  description,
  runs,
  rating,
  onClick,
}) => {
  const renderStars = () => {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 !== 0;
    const stars = [];

    for (let i = 0; i < 5; i++) {
      if (i < fullStars) {
        stars.push(<StarFilledIcon key={i} className="text-black" />);
      } else if (i === fullStars && hasHalfStar) {
        stars.push(<StarIcon key={i} className="text-black" />);
      } else {
        stars.push(<StarIcon key={i} className="text-black" />);
      }
    }

    return stars;
  };

  return (
    <div
      className={`
        w-screen px-2 pb-2 pt-4 gap-3
        md:w-[41.875rem] md:h-[37.188rem] md:px-[1.5625rem] md:pb-[0.9375rem] md:pt-[2.1875rem] md:gap-5 rounded-xl
        text-sm md:text-xl tracking-tight font-neue text-[#272727]
        inline-flex cursor-pointer flex-col items-start justify-between
        border border-black/10 bg-[#f9f9f9] transition-shadow duration-300 hover:shadow-lg
      `}
      onClick={onClick}
    >
      <div className="flex flex-col items-start justify-start self-stretch">
        <div className="text-2xl md:text-4xl font-medium self-stretch  ">
          {agentName}
        </div>
        <div className="font-normal self-stretch text-[#878787]">
          by {creatorName}
        </div>
      </div>
      <div className="w-full md:w-[33.75rem] max-h-18 font-normal flex-grow text-clip text-[#282828]">
        {description.length > 170 ? `${description.slice(0, 170)}...` : description}
      </div>
      <div className="gap-3 flex flex-col items-start justify-end  self-stretch">
        <div className="w-full aspect-[540/245] relative">
          <Image
            src={agentImage}
            alt={`${agentName} preview`}
            layout="fill"
            objectFit="cover"
            className="rounded-xl "
          />
        </div>
        <div className="flex items-center justify-between self-stretch">
          <div>
            <span className="font-medium">
              {runs.toLocaleString()}+
            </span>
            <span className="font-normal">
              {" "}
              runs
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className=" font-normal  ">
              {rating.toFixed(1)}
            </div>
            <div className="flex items-center justify-start gap-px">
              {renderStars()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
