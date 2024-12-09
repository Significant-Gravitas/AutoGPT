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
      className={`group h-[755px] w-[440px] px-[22px] pb-5 pt-[30px] ${backgroundColor} inline-flex flex-col items-start justify-start gap-7 rounded-[26px] transition-all duration-200 hover:brightness-95 dark:bg-neutral-800`}
      onClick={onClick}
      data-testid="featured-store-card"
    >
      <div className="flex h-[188px] flex-col items-start justify-start gap-3 self-stretch">
        <h2 className="font-poppins self-stretch text-[35px] font-medium leading-10 text-neutral-900 dark:text-neutral-100">
          {agentName}
        </h2>
        <div className="font-lead self-stretch text-xl font-normal leading-7 text-neutral-800 dark:text-neutral-200">
          {subHeading}
        </div>
      </div>

      <div className="flex h-[489px] flex-col items-start justify-start gap-[18px] self-stretch">
        <div className="font-lead self-stretch text-xl font-normal leading-7 text-neutral-800 dark:text-neutral-200">
          by {creatorName}
        </div>

        <div className="relative h-[397px] self-stretch">
          <Image
            src={agentImage}
            alt={`${agentName} preview`}
            layout="fill"
            objectFit="cover"
            className="rounded-xl transition-opacity duration-200 group-hover:opacity-0"
          />
          <div className="absolute inset-0 overflow-y-auto rounded-xl bg-white p-4 opacity-0 transition-opacity duration-200 group-hover:opacity-100 dark:bg-neutral-700">
            <div className="font-geist text-base font-normal leading-normal text-neutral-800 dark:text-neutral-200">
              {description}
            </div>
          </div>
          {creatorImage && (
            <div className="absolute left-[8.74px] top-[313px] h-[74px] w-[74px] overflow-hidden rounded-full transition-opacity duration-200 group-hover:opacity-0">
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

        <div className="inline-flex items-center justify-between self-stretch">
          <div className="font-large-geist text-lg font-semibold leading-7 text-neutral-800 dark:text-neutral-200">
            {runs.toLocaleString()} runs
          </div>
          <div className="flex items-center justify-start gap-[5px]">
            <div className="font-large-geist text-lg font-semibold leading-7 text-neutral-800 dark:text-neutral-200">
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
