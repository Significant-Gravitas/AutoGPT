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
      className="w-full min-w-80 max-w-md space-y-2 rounded-3xl bg-white p-2 pb-3 hover:bg-gray-50"
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
      {/* First Section: Image with Avatar */}
      <div className="relative aspect-[2/1.2] w-full overflow-hidden rounded-3xl md:aspect-[1.78/1]">
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
          <div className="absolute bottom-4 left-4 rounded-full border border-zinc-200">
            <Avatar className="h-16 w-16">
              {avatarSrc && (
                <AvatarImage
                  src={avatarSrc}
                  alt={`${creatorName || agentName} creator avatar`}
                />
              )}
              <AvatarFallback size={64}>
                {(creatorName || agentName).charAt(0)}
              </AvatarFallback>
            </Avatar>
          </div>
        )}
      </div>

      <div className="flex w-full flex-1 flex-col">
        {/* Second Section: Agent Name and Creator Name */}
        <div className="flex w-full flex-col px-1.5">
          <h3 className="line-clamp-2 h-12 font-sans text-base font-medium text-zinc-800 dark:text-neutral-100">
            {agentName}
          </h3>
          {!hideAvatar && creatorName && (
            <p className="truncate font-sans text-sm font-normal text-zinc-600 dark:text-neutral-400">
              by {creatorName}
            </p>
          )}
        </div>

        <div className="flex h-18 w-full flex-col px-1.5 pt-2">
          <p className="line-clamp-3 font-sans text-sm font-normal text-zinc-500 dark:text-neutral-400">
            {description}
          </p>
        </div>

        {/* Fourth Section: Stats Row - aligned to bottom */}
        <div className="mt-2.5 flex items-center justify-between px-1.5 pt-2">
          <div className="font-sans text-sm font-medium text-zinc-800 dark:text-neutral-200">
            {runs.toLocaleString()} runs
          </div>
          <div className="flex items-center gap-2">
            <span className="font-sans text-sm font-medium text-zinc-800 dark:text-neutral-200">
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
  );
};
