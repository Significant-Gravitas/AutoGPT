import * as React from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { StarRatingIcons } from "@/components/ui/icons";

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
      className="inline-flex h-auto min-h-[500px] w-full max-w-[440px] flex-col items-start justify-between rounded-[26px] bg-violet-100 p-4 sm:h-[632px] sm:w-[440px] sm:p-6"
      role="article"
      aria-label={`Creator profile for ${username}`}
    >
      <div className="flex w-full flex-col items-start justify-start gap-3.5 sm:h-[218px]">
        <Avatar className="h-[100px] w-[100px] sm:h-[130px] sm:w-[130px]">
          <AvatarImage src={avatarSrc} alt={`${username}'s avatar`} />
          <AvatarFallback className="h-[100px] w-[100px] sm:h-[130px] sm:w-[130px]">
            {username.charAt(0)}
          </AvatarFallback>
        </Avatar>
        <div className="flex w-full flex-col items-start justify-start gap-1.5">
          <div className="w-full font-poppins text-2xl sm:text-[35px] font-medium leading-8 sm:leading-10 text-neutral-900">
            {username}
          </div>
          <div className="w-full font-neue text-lg sm:text-xl font-normal leading-6 sm:leading-7 text-neutral-800">
            @{handle}
          </div>
        </div>
      </div>

      <div className="flex w-full flex-col items-start justify-start gap-6 sm:gap-[50px] my-4">
        <div className="flex w-full flex-col items-start justify-start gap-3">
          <div className="h-px w-full bg-neutral-700" />
          <div className="flex flex-col items-start justify-start gap-2.5">
            <div className="w-full font-neue text-base font-medium leading-normal text-neutral-800">
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
                  className="flex items-center justify-center gap-2.5 rounded-[34px] border border-neutral-600 px-5 py-3"
                  role="listitem"
                >
                  <div className="font-neue text-base font-normal leading-normal text-neutral-800">
                    {category}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex w-full flex-col items-start justify-start gap-3">
          <div className="h-px w-full bg-neutral-700" />
          <div className="flex w-full flex-col sm:flex-row justify-between items-start gap-4 sm:gap-0">
            <div className="w-full sm:w-[164px] flex flex-col items-start justify-start gap-2.5">
              <div className="w-full font-neue text-base font-medium leading-normal text-neutral-800">
                Average rating
              </div>
              <div className="inline-flex items-center gap-2">
                <div className="font-neue text-lg font-semibold leading-7 text-neutral-800">
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
            <div className="w-full sm:w-[164px] flex flex-col items-start justify-start gap-2.5">
              <div className="w-full font-neue text-base font-medium leading-normal text-neutral-800">
                Number of runs
              </div>
              <div className="font-neue text-lg font-semibold leading-7 text-neutral-800">
                {new Intl.NumberFormat().format(totalRuns)} runs
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};