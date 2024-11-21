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
}) => {
  const handleClick = () => {
    onClick();
  };

  return (
    <div
      className="flex h-96 w-64 flex-col rounded-xl pb-2 transition-shadow duration-300 hover:shadow-lg sm:w-64 md:w-80 xl:w-110"
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
      <div className="relative h-48 w-full">
        <Image
          src={agentImage}
          alt={`${agentName} preview image`}
          fill
          sizes="192px"
          className="rounded-xl object-cover"
        />
      </div>
      <div className="-mt-8 flex flex-col px-4">
        {!hideAvatar ? (
          <Avatar className="mb-2 h-16 w-16">
            <AvatarImage src={avatarSrc} alt={`${agentName} creator avatar`} />
            <AvatarFallback
              className="h-16 w-16"
              role="img"
              aria-label={`${agentName} creator initial`}
            >
              {agentName.charAt(0)}
            </AvatarFallback>
          </Avatar>
        ) : (
          <div className="h-16" aria-hidden="true" />
        )}
        <h2 className="mb-1 font-neue text-xl font-bold tracking-tight text-neutral-900">
          {agentName}
        </h2>
        <div className="mb-4 flex items-center justify-between">
          <div className="font-neue text-base font-medium tracking-tight text-neutral-900">
            {runs.toLocaleString()}+ runs
          </div>
          <div className="flex items-center">
            <div className="mr-2 font-neue text-base font-medium tracking-tight text-neutral-900">
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
        <p className="font-neue text-base font-normal leading-[21px] tracking-tight text-neutral-900">
          {description}
        </p>
      </div>
    </div>
  );
};
