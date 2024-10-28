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
      className="group inline-flex h-[775px] w-[440px] cursor-pointer flex-col items-start justify-start gap-7 rounded-[26px] bg-neutral-100 px-[22px] pb-5 pt-[30px] font-neue text-sm tracking-tight transition-shadow duration-300 hover:shadow-lg"
      onClick={onClick}
      data-testid="featured-store-card"
    >
      <div className="flex h-[188px] w-full flex-col items-start justify-start gap-3 group-hover:h-fit">
        <h2 className="font-['Poppins'] text-[35px] font-medium leading-10 text-neutral-900">
          {agentName}
        </h2>
        <p className="font-['Geist'] text-xl font-normal leading-7 text-neutral-800">
          {subHeading}
        </p>
      </div>

      <div className="relative w-full flex-1">
        {/* Default view */}
        <div className="absolute inset-0 flex flex-col gap-[18px] group-hover:opacity-0">
          <p className="font-['Geist'] text-xl font-normal leading-7 text-neutral-800">
            by {creatorName}
          </p>

          <div className="relative aspect-square w-full flex-1">
            <Image
              src={agentImage}
              alt={`${agentName} preview`}
              width={396}
              height={396}
              className="aspect-square rounded-xl object-cover"
            />
          </div>
        </div>

        {/* Hovered view */}
        <div className="absolute inset-0 flex flex-col gap-[18px] opacity-0 group-hover:opacity-100">
          <p className="font-['Geist'] text-xl font-normal leading-7 text-neutral-800">
            by {creatorName}
          </p>

          <div className="flex-1 overflow-y-auto rounded-xl py-4">
            <p className="font-['Geist'] text-xl font-normal leading-7 text-neutral-800">
              {description}
            </p>
          </div>
        </div>
      </div>

      <div className="flex w-full items-center justify-between">
        <div className="font-['Inter'] text-lg font-semibold leading-7 text-neutral-800">
          {runs.toLocaleString()} runs
        </div>
        <div className="flex items-center gap-[5px]">
          <div className="font-['Inter'] text-lg font-semibold leading-7 text-neutral-800">
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
