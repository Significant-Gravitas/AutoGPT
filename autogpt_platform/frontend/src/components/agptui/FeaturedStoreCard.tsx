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
}) => {
  return (
    <div
      className="group flex h-[90%] w-[90%] cursor-pointer flex-col items-start justify-start gap-3 rounded-[26px] bg-neutral-200 px-[22px] pb-5 pt-[30px] font-neue text-sm tracking-tight transition-shadow duration-300 hover:shadow-lg md:max-h-[705px] md:gap-5 lg:w-[440px]"
      onClick={onClick}
      data-testid="featured-store-card"
    >
      <div
        className={`flex h-[216px] flex-col items-start justify-start gap-3 self-stretch md:h-[188px]`}
      >
        <div
          className={`self-stretch font-['Poppins'] text-[35px] font-medium leading-10 text-neutral-900`}
        >
          {agentName}
        </div>
        <div
          className={`self-stretch font-['Geist'] text-xl font-normal leading-7 text-neutral-800`}
        >
          {subHeading}
        </div>
      </div>

      <div
        className={`flex h-[489px] flex-col items-start justify-start gap-[18px] self-stretch`}
      >
        <div
          className={`self-stretch font-['Geist'] text-xl font-normal leading-7 text-neutral-800`}
        >
          by {creatorName}
        </div>

        <div className="relative h-[397px] w-full">
          {/* Image Container */}
          <div className={` ${"group-hover:hidden"} relative h-full w-full`}>
            <div
              className={`relative h-[397px] w-[346px] md:h-[397px] md:w-[396px] lg:h-[397px] lg:w-[456px]`}
            >
              <Image
                src={agentImage}
                alt={`${agentName} preview`}
                layout="fill"
                objectFit="cover"
                className="rounded-xl"
              />
              {creatorImage && (
                <Image
                  src={creatorImage}
                  alt={`${creatorName} image`}
                  width={74}
                  height={74}
                  className={`absolute left-[8.74px] top-[313px] h-[74px] w-[74px] rounded-full md:left-[10px] lg:left-[11.52px]`}
                />
              )}
            </div>
          </div>

          {/* Description Container */}
          <div
            className={` ${"hidden group-hover:flex"} absolute inset-0 flex flex-col overflow-y-auto rounded-xl bg-white bg-opacity-90 p-4`}
          >
            <p
              className={`font-['Geist'] text-lg font-normal leading-7 text-neutral-800`}
            >
              {description}
            </p>
          </div>
        </div>

        <div className={`flex items-center justify-between self-stretch`}>
          <div
            className={`font-['Inter'] text-lg font-semibold leading-7 text-neutral-800`}
          >
            {runs.toLocaleString()} runs
          </div>
          <div className={`flex items-center gap-[5px]`}>
            <div
              className={`font-['Inter'] text-lg font-semibold leading-7 text-neutral-800`}
            >
              {rating.toFixed(1)}
            </div>
            <div className={`relative h-4 w-[84px]`}>
              {StarRatingIcons(rating)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
