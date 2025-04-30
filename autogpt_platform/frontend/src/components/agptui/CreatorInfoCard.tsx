import * as React from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { StarRatingIcons } from "@/components/ui/icons";
import { Separator } from "../ui/separator";

interface CreatorInfoCardProps {
  username: string;
  handle: string;
  avatarSrc: string;
  categories: string[];
  averageRating: number;
  totalRuns: number;
}

export const CreatorInfoCard: React.FC<CreatorInfoCardProps> = ({
  username,
  handle,
  avatarSrc,
  categories,
  averageRating,
  totalRuns,
}) => {
  return (
    <div
      className="inline-flex h-auto min-h-[32rem] w-full max-w-[440px] flex-col items-start justify-between rounded-[26px] bg-violet-100 p-4 dark:bg-violet-900 sm:min-h-[40rem] sm:w-[440px] sm:p-6"
      role="article"
      aria-label={`Creator profile for ${username}`}
    >
      {/* Avatar + Basic Info */}
      <div className="flex w-full flex-col items-start justify-start gap-3.5">
        <Avatar className="h-[100px] w-[100px] sm:h-[130px] sm:w-[130px]">
          <AvatarImage
            width={130}
            height={130}
            src={avatarSrc}
            alt={`${username}'s avatar`}
          />
          <AvatarFallback
            size={130}
            className="h-[100px] w-[100px] sm:h-[130px] sm:w-[130px]"
          >
            {username.charAt(0)}
          </AvatarFallback>
        </Avatar>
        <div className="flex w-full flex-col items-start justify-start gap-1.5">
          <div className="w-full font-poppins text-2xl font-medium text-neutral-900 dark:text-neutral-100 sm:text-4xl">
            {username}
          </div>
          <div className="w-full font-sans text-lg font-normal text-neutral-800 dark:text-neutral-200 sm:text-xl">
            @{handle}
          </div>
        </div>
      </div>

      <div className="my-4 flex w-full flex-col items-start justify-start gap-6 sm:gap-12">
        {/* Categories */}
        <div className="flex w-full flex-col items-start justify-start gap-3">
          <Separator className="bg-neutral-700" />
          <div className="flex flex-col items-start justify-start gap-2.5">
            <div className="w-full font-sans text-sm font-medium text-neutral-800 dark:text-neutral-200 sm:text-base">
              Top categories
            </div>
            <div
              className="flex flex-wrap items-center gap-2.5"
              role="list"
              aria-label="Categories"
            >
              {categories.map((category, index) => (
                <div
                  key={index}
                  className="flex items-center justify-center gap-2.5 rounded-[34px] border border-neutral-600 px-4 py-3 dark:border-neutral-400"
                  role="listitem"
                >
                  <div className="font-sans text-sm font-normal text-neutral-800 dark:text-neutral-200 sm:text-base">
                    {category}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Ratings */}
        <div className="flex w-full flex-col items-start justify-start gap-3">
          <Separator className="bg-neutral-700" />
          <div className="flex w-full flex-col items-center justify-between gap-4 sm:flex-row sm:gap-0">
            {/* Average Rating */}
            <div className="flex w-full flex-col items-start justify-start gap-2.5">
              <div className="w-full font-sans text-sm font-medium leading-normal text-neutral-800 dark:text-neutral-200 sm:text-base">
                Average rating
              </div>
              <div className="inline-flex items-center gap-2">
                <div className="font-sans text-base font-semibold text-neutral-800 dark:text-neutral-200 sm:text-lg">
                  {averageRating.toFixed(1)}
                </div>
                <div
                  className="flex items-center gap-px"
                  role="img"
                  aria-label={`Rating: ${averageRating} out of 5 stars`}
                >
                  {StarRatingIcons(averageRating)}
                </div>
              </div>
            </div>
            {/* Number of runs */}
            <div className="flex w-full flex-col items-start justify-start gap-2.5">
              <div className="w-full font-sans text-sm font-medium leading-normal text-neutral-800 dark:text-neutral-200 sm:text-base">
                Number of runs
              </div>
              <div className="font-sans text-sm font-semibold text-neutral-800 dark:text-neutral-200 sm:text-lg">
                {new Intl.NumberFormat().format(totalRuns)} runs
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
