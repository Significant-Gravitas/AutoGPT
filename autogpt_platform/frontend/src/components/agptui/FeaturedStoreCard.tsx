import * as React from "react";
import Image from "next/image";
import { StarRatingIcons } from "@/components/ui/icons";
interface FeaturedStoreCardProps {
  agentName: string;
  subHeading: string;
  agentImage: string;
  creatorName: string;
  description: string;
  runs: number;
  rating: number;
  onClick: () => void;
}

export const FeaturedStoreCard: React.FC<FeaturedStoreCardProps> = ({
  agentName,
  subHeading,
  agentImage,
  creatorName,
  description,
  runs,
  rating,
  onClick,
}) => {
  return (
    <div
      className="group flex h-[90%] w-[90%] cursor-pointer flex-col items-start justify-start gap-3 rounded-[26px] bg-neutral-300 px-[22px] pb-5 pt-[30px] font-neue text-sm tracking-tight transition-shadow duration-300 hover:shadow-lg md:max-h-[705px] md:w-[440px] md:gap-5"
      onClick={onClick}
      data-testid="featured-store-card"
    >
      <div className="flex w-full flex-col items-start justify-start gap-1 md:gap-3">
        <h2 className="font-['Poppins'] text-2xl font-medium leading-tight text-neutral-900 md:text-[35px] md:leading-10">
          {agentName}
        </h2>
        <p className="font-['Geist'] text-lg font-normal leading-7 text-neutral-800 md:text-xl">
          {subHeading}
        </p>
        <p className="font-['Geist'] text-lg font-normal leading-7 text-neutral-800 md:text-xl">
          by {creatorName}
        </p>
      </div>

      <div className="w-full flex-1">
        <div className="flex flex-col gap-[18px] transition-opacity duration-300 group-hover:hidden">
          <div className="relative w-full">
            <Image
              src={agentImage}
              alt={`${agentName} preview`}
              width={396}
              height={396}
              className="aspect-square w-full rounded-xl object-cover"
            />
          </div>
        </div>

        <div className="hidden aspect-square w-full flex-col gap-[18px] transition-opacity duration-300 group-hover:flex">
          <div className="flex-1 overflow-y-auto rounded-xl py-4">
            <p className="font-['Geist'] text-lg font-normal leading-7 text-neutral-800 md:text-xl">
              {description}
            </p>
          </div>
        </div>
      </div>

      <div className="flex w-full items-center justify-between">
        <div className="font-['Inter'] text-base font-semibold leading-7 text-neutral-800 md:text-lg">
          {runs.toLocaleString()} runs
        </div>
        <div className="flex items-center gap-[5px]">
          <div className="font-['Inter'] text-base font-semibold leading-7 text-neutral-800 md:text-lg">
            {rating.toFixed(1)}
          </div>
          <div className="flex w-[84px] items-center justify-start gap-px">
            {StarRatingIcons(rating)}
          </div>
        </div>
      </div>
    </div>
  );
};
