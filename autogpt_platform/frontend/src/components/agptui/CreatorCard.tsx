import * as React from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import Image from "next/image";

interface CreatorCardProps {
  creatorName: string;
  creatorImage: string;
  bio: string;
  agentsUploaded: number;
  onClick: () => void;
  avatarSrc: string;
}

export const CreatorCard: React.FC<CreatorCardProps> = ({
  creatorName,
  creatorImage,
  bio,
  agentsUploaded,
  onClick,
  avatarSrc,
}) => {
  const avatarSizeClasses = "h-16 w-16";

  return (
    <div
      className="flex h-96 w-[13.125rem] flex-col rounded-xl transition-shadow duration-300 hover:shadow-lg"
      onClick={onClick}
      data-testid="creator-card"
    >
      <div className="relative aspect-[210/238] w-full">
        <Image
          src={creatorImage}
          alt={creatorName}
          layout="fill"
          objectFit="cover"
          className="rounded-xl"
        />
      </div>
      <div className="relative -mt-20 ml-4">
        <Avatar className={avatarSizeClasses}>
          <AvatarImage src={avatarSrc} alt={`${creatorName}'s avatar`} />
          <AvatarFallback className={avatarSizeClasses}>
            {creatorName.charAt(0)}
          </AvatarFallback>
        </Avatar>
      </div>
      <div className="mt-8 font-neue text-xl font-bold tracking-tight text-[#272727]">
        {creatorName}
      </div>
      <div className="mt-2 line-clamp-3 w-full font-neue text-base font-normal leading-[21px] tracking-tight text-[#282828]">
        {bio}
      </div>
      <div className="mt-auto font-neue text-base font-medium tracking-tight text-[#272727]">
        {agentsUploaded} agents uploaded
      </div>
    </div>
  );
};
