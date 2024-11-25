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
      className="w-full max-w-[434px] rounded-[26px] flex-col justify-start items-start gap-2.5 inline-flex cursor-pointer hover:shadow-lg transition-all duration-300"
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
      {/* Header Image Section with Avatar */}
      <div className="relative w-full h-[200px] rounded-[20px] overflow-hidden">
        <Image
          src={agentImage}
          alt={`${agentName} preview image`}
          fill
          className="object-cover"
          priority
        />
        {!hideAvatar && (
          <div className="absolute left-4 bottom-4">
            <Avatar className="h-16 w-16 border-2 border-white">
              <AvatarImage 
                src={avatarSrc} 
                alt={`${creatorName || agentName} creator avatar`}
              />
              <AvatarFallback>
                {(creatorName || agentName).charAt(0)}
              </AvatarFallback>
            </Avatar>
          </div>
        )}
      </div>

      {/* Content Section */}
      <div className="w-full px-1 py-4">
        {/* Title and Creator */}
        <h3 className="text-[#272727] text-2xl font-semibold font-['Poppins'] leading-tight mb-2">
          {agentName}
        </h3>
        {!hideAvatar && creatorName && (
          <p className="text-neutral-600 text-base font-normal font-['Geist'] mb-4">
            by {creatorName}
          </p>
        )}

        {/* Description */}
        <p className="text-neutral-600 text-base font-normal font-['Geist'] leading-normal line-clamp-3 mb-4">
          {description}
        </p>

        {/* Stats Row */}
        <div className="flex justify-between items-center">
          <div className="text-neutral-800 text-lg font-semibold font-['Geist']">
            {runs.toLocaleString()} runs
          </div>
          <div className="flex items-center gap-2">
            <span className="text-neutral-800 text-lg font-semibold font-['Geist']">
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
