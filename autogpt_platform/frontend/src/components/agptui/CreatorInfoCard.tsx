import * as React from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { StarRatingIcons } from "@/components/ui/icons";
import { Separator } from "../ui/separator";
import { Chip } from "./Chip";

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
      className="h-auto w-full max-w-md space-y-6 overflow-hidden rounded-[26px] bg-violet-100 px-5 pb-7 pt-6 dark:bg-violet-900 sm:w-[27rem]"
      role="article"
      aria-label={`Creator profile for ${username}`}
    >
      {/* Avatar + Basic Info */}
      <div className="flex w-full flex-col items-start justify-start gap-3.5">
        <Avatar className="h-[100px] w-[100px]">
          <AvatarImage
            width={100}
            height={100}
            src={avatarSrc}
            alt={`${username}'s avatar`}
          />
          <AvatarFallback size={100} className="h-[100px] w-[100px]">
            {username.charAt(0)}
          </AvatarFallback>
        </Avatar>
        <div className="flex w-full flex-col items-start justify-start gap-1.5">
          <div className="w-full font-poppins text-3xl font-medium text-zinc-800 dark:text-zinc-100">
            {username}
          </div>
          <div className="w-full font-sans text-base font-normal text-zinc-800 dark:text-zinc-200">
            @{handle}
          </div>
        </div>
      </div>

      <Separator className="bg-zinc-300" />

      <div className="flex flex-col items-start justify-start gap-2.5">
        <div className="w-full font-sans text-sm font-medium text-zinc-800 dark:text-zinc-200 sm:text-base">
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
              className="flex items-center justify-center gap-2.5"
              role="listitem"
            >
              <Chip className="bg-transparent">{category}</Chip>
            </div>
          ))}
        </div>
      </div>

      <Separator className="bg-zinc-300" />

      <div className="flex w-full flex-col items-center justify-between gap-4 sm:flex-row sm:gap-0">
        {/* Average Rating */}
        <div className="flex w-full flex-col items-start justify-start gap-2.5">
          <div className="w-full font-sans text-sm font-medium leading-normal text-zinc-800 dark:text-zinc-200 sm:text-base">
            Average rating
          </div>
          <div className="inline-flex items-center gap-2">
            <div className="font-sans text-sm font-medium text-zinc-800 dark:text-zinc-200">
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
          <div className="w-full font-sans text-sm font-medium leading-normal text-zinc-800 dark:text-zinc-200 sm:text-base">
            Number of runs
          </div>
          <div className="font-sans text-sm font-medium text-zinc-800 dark:text-zinc-200 sm:text-base">
            {new Intl.NumberFormat().format(totalRuns)} runs
          </div>
        </div>
      </div>
    </div>
  );
};
