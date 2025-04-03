import * as React from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import Image from "next/image";
import { StarRatingIcons } from "@/components/ui/icons";

interface StoreCardProps {
  agentName: string;
  agentImage: string;
  description: string;
  runs: number;
  rating: number;
  onClick: () => void;
  avatarSrc: string;
  hideAvatar?: boolean;
  creatorName?: string;
}

export const StoreCard: React.FC<StoreCardProps> = ({
  agentName,
  agentImage,
  description,
  runs,
  rating,
  onClick,
  avatarSrc,
  hideAvatar = false,
  creatorName,
}) => {
  const handleClick = () => {
    onClick();
  };

  return (
    <div
      className="inline-flex h-full w-full max-w-[434px] cursor-pointer flex-col items-start justify-between rounded-[26px] bg-white transition-all duration-300 hover:shadow-lg dark:bg-transparent dark:hover:shadow-gray-700"
      onClick={handleClick}
      data-testid="store-card"
      role="button"
      tabIndex={0}
      aria-label={`${agentName} agent card`}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          handleClick();
        }
      }}
    >
      <div className="flex h-full w-full flex-col">
        {/* Header Image Section with Avatar - fixed aspect ratio */}
        <div className="relative aspect-[2.17/2] w-full overflow-hidden rounded-[20px]">
          {agentImage && (
            <Image
              src={agentImage}
              alt={`${agentName} preview image`}
              fill
              className="object-cover"
              priority
            />
          )}
          {!hideAvatar && (
            <div className="absolute bottom-4 left-4">
              <Avatar className="h-16 w-16 border-2 border-white dark:border-gray-800">
                {avatarSrc && (
                  <AvatarImage
                    src={avatarSrc}
                    alt={`${creatorName || agentName} creator avatar`}
                  />
                )}
                <AvatarFallback>
                  {(creatorName || agentName).charAt(0)}
                </AvatarFallback>
              </Avatar>
            </div>
          )}
        </div>

        {/* Content Section */}
        <div className="flex h-full w-full flex-col px-2 py-4">
          <div className="flex-grow">
            {/* Title and Creator */}
            <h3 className="mb-0.5 line-clamp-2 h-[60px] font-poppins text-2xl font-semibold leading-tight text-[#272727] dark:text-neutral-100">
              {agentName}
            </h3>
            {!hideAvatar && creatorName && (
              <p className="font-lead mb-2.5 truncate text-base font-normal text-neutral-600 dark:text-neutral-400">
                by {creatorName}
              </p>
            )}
            {/* Description */}
            <p className="font-geist mb-4 line-clamp-3 h-[72px] text-base font-normal leading-normal text-neutral-600 dark:text-neutral-400">
              {description}
            </p>
          </div>

          {/* Stats Row - fixed to bottom */}
          <div className="mt-auto flex items-center justify-between">
            <div className="font-geist text-lg font-semibold text-neutral-800 dark:text-neutral-200">
              {runs.toLocaleString()} runs
            </div>
            <div className="flex items-center gap-2">
              <span className="font-geist text-lg font-semibold text-neutral-800 dark:text-neutral-200">
                {rating.toFixed(1)}
              </span>
              <div
                className="inline-flex items-center"
                role="img"
                aria-label={`Rating: ${rating.toFixed(1)} out of 5 stars`}
              >
                {StarRatingIcons(rating)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
