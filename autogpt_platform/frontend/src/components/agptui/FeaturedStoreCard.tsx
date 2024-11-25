import * as React from "react";
import Image from "next/image";
import { StarRatingIcons } from "@/components/ui/icons";

interface FeaturedStoreCardProps {
  agentName: string;
  subHeading: string;
  agentImage: string;
  creatorImage?: string;
  creatorName: string;
  description: string; // Added description prop
  runs: number;
  rating: number;
  onClick: () => void;
  backgroundColor: string;
}

export const FeaturedStoreCard: React.FC<FeaturedStoreCardProps> = ({
  agentName,
  subHeading,
  agentImage,
  creatorImage,
  creatorName,
  description,
  runs,
  rating,
  onClick,
  backgroundColor,
}) => {
  return (
    <div
      className={`group w-[440px] h-[755px] px-[22px] pt-[30px] pb-5 ${backgroundColor} hover:brightness-95 rounded-[26px] flex-col justify-start items-start gap-7 inline-flex transition-all duration-200`}
      onClick={onClick}
      data-testid="featured-store-card"
    >
      <div className="self-stretch h-[188px] flex-col justify-start items-start gap-3 flex">
        <div className="self-stretch text-neutral-900 text-[35px] font-medium font-['Poppins'] leading-10">
          {agentName}
        </div>
        <div className="self-stretch text-neutral-800 text-xl font-normal font-['Geist'] leading-7">
          {subHeading}
        </div>
      </div>

      <div className="self-stretch h-[489px] flex-col justify-start items-start gap-[18px] flex">
        <div className="self-stretch text-neutral-800 text-xl font-normal font-['Geist'] leading-7">
          by {creatorName}
        </div>
        
        <div className="relative self-stretch h-[397px]">
          <Image
            src={agentImage}
            alt={`${agentName} preview`}
            layout="fill"
            objectFit="cover"
            className="rounded-xl group-hover:opacity-0 transition-opacity duration-200"
          />
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-200 rounded-xl bg-white p-4 overflow-y-auto">
            <div className="text-neutral-800 text-base font-normal font-['Geist'] leading-normal">
              {description}
            </div>
          </div>
          {creatorImage && (
            <div className="absolute left-[8.74px] top-[313px] w-[74px] h-[74px] overflow-hidden rounded-full group-hover:opacity-0 transition-opacity duration-200">
              <Image
                src={creatorImage}
                alt={`${creatorName} image`}
                layout="fill"
                className="object-cover"
                priority
              />
            </div>
          )}
        </div>

        <div className="self-stretch justify-between items-center inline-flex">
          <div className="text-neutral-800 text-lg font-semibold font-['Inter'] leading-7">
            {runs.toLocaleString()} runs
          </div>
          <div className="justify-start items-center gap-[5px] flex">
            <div className="text-neutral-800 text-lg font-semibold font-['Inter'] leading-7">
              {rating.toFixed(1)}
            </div>
            <div 
              className="inline-flex items-center justify-start gap-px"
              role="img"
              aria-label={`Rating: ${rating.toFixed(1)} out of 5 stars`}
            >
              {StarRatingIcons(rating)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
