import * as React from "react";
import Image from "next/image";
import { StarRatingIcons } from "../ui/icons";
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
  return (
    <div
      className={`inline-flex w-[calc(100vw-4rem)] cursor-pointer flex-col items-start justify-between gap-3 rounded-xl border border-black/10 bg-[#f9f9f9] px-2 pb-2 pt-4 font-neue text-sm tracking-tight text-[#272727] transition-shadow duration-300 hover:shadow-lg md:h-[37.188rem] md:w-[41.875rem] md:gap-5 md:px-[1.5625rem] md:pb-[0.9375rem] md:pt-[2.1875rem] md:text-xl lg:basis-full`}
      onClick={onClick}
      data-testid="featured-store-card"
    >
      <div className="flex flex-col items-start justify-start self-stretch">
        <div className="self-stretch text-2xl font-medium md:text-4xl">
          {agentName}
        </div>
        <div className="self-stretch font-normal text-[#737373]">
          by {creatorName}
        </div>
      </div>
      <div className="max-h-18 w-full flex-grow text-clip font-normal text-[#282828] md:w-[33.75rem]">
        {description.length > 170
          ? `${description.slice(0, 170)}...`
          : description}
      </div>
      <div className="flex flex-col items-start justify-end gap-3 self-stretch">
        <div className="relative aspect-[540/245] w-full">
          <Image
            src={agentImage}
            alt={`${agentName} preview`}
            layout="fill"
            objectFit="cover"
            className="rounded-xl"
          />
        </div>
        <div className="flex items-center justify-between self-stretch">
          <div>
            <span className="font-medium">{runs.toLocaleString()}+</span>
            <span className="font-normal"> runs</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="font-normal">{rating.toFixed(1)}</div>
            <div className="flex items-center justify-start gap-px">
              {StarRatingIcons(rating)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
