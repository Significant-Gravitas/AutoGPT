"use client";

import { StarRatingIcons } from "@/components/__legacy__/ui/icons";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import Image from "next/image";
import { useState } from "react";
import { AddToLibraryButton } from "../AddToLibraryButton/AddToLibraryButton";

interface Props {
  agentName: string;
  agentImage: string;
  description: string;
  runs: number;
  rating: number;
  onClick: () => void;
  avatarSrc: string;
  hideAvatar?: boolean;
  creatorName?: string;
  creatorSlug?: string;
  agentSlug?: string;
  agentGraphID?: string;
}

export function StoreCard({
  agentName,
  agentImage,
  description,
  runs,
  rating,
  onClick,
  avatarSrc,
  hideAvatar = false,
  creatorName,
  creatorSlug,
  agentSlug,
  agentGraphID,
}: Props) {
  const [imageError, setImageError] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);

  const handleClick = () => {
    onClick();
  };

  return (
    <div
      className="relative flex h-[26.5rem] w-full max-w-md cursor-pointer flex-col items-start rounded-2xl border border-border/50 bg-background p-4 shadow-md transition-all duration-300 hover:shadow-lg"
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
      {/* First Section: Image */}
      <div className="relative aspect-[2/1.2] w-full overflow-hidden rounded-xl md:aspect-[2.17/1]">
        {agentImage && !imageError ? (
          <>
            {!imageLoaded && (
              <Skeleton className="absolute inset-0 rounded-xl" />
            )}
            <Image
              src={agentImage}
              alt={`${agentName} preview image`}
              fill
              className="object-cover"
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageError(true)}
            />
          </>
        ) : (
          <div className="absolute inset-0 rounded-xl bg-violet-50" />
        )}
      </div>

      <div className="mt-3 flex w-full flex-1 flex-col">
        {/* Second Section: Agent Name and Creator Name */}
        <div className="flex w-full min-w-0 flex-col gap-1">
          <OverflowText
            value={agentName}
            variant="h4"
            className="text-xl leading-tight"
          />
          {!hideAvatar && creatorName && (
            <div className="mb-2 mt-2 flex items-center gap-2">
              <Avatar className="h-6 w-6 shrink-0">
                {avatarSrc && (
                  <AvatarImage
                    src={avatarSrc}
                    alt={`${creatorName} creator avatar`}
                  />
                )}
                <AvatarFallback size={32}>
                  {creatorName.charAt(0)}
                </AvatarFallback>
              </Avatar>
              <Text variant="body-medium" className="truncate">
                by {creatorName}
              </Text>
            </div>
          )}
        </div>

        {/* Third Section: Description */}
        <div className="mt-2.5 flex w-full flex-col">
          <Text variant="body" className="line-clamp-3 leading-normal">
            {description}
          </Text>
        </div>
      </div>

      {/* Stats */}
      <Text variant="body" className="absolute bottom-4 left-4 text-zinc-500">
        {runs === 0 ? "No runs" : `${runs.toLocaleString()} runs`}
      </Text>
      {rating >= 1 && (
        <div className="absolute bottom-4 right-4 flex items-center gap-2">
          <span className="text-lg font-semibold text-neutral-800">
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
      )}
      {creatorSlug && agentSlug && agentGraphID && (
        <div className="absolute bottom-2 right-0">
          <AddToLibraryButton
            creatorSlug={creatorSlug}
            agentSlug={agentSlug}
            agentName={agentName}
            agentGraphID={agentGraphID}
          />
        </div>
      )}
    </div>
  );
}
