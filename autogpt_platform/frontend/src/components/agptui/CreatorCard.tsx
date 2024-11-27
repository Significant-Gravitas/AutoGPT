import * as React from "react";
import Image from "next/image";

const BACKGROUND_COLORS = [
  "bg-amber-100", // #fef3c7
  "bg-violet-100", // #ede9fe
  "bg-green-100", // #dcfce7
  "bg-blue-100", // #dbeafe
];

interface CreatorCardProps {
  creatorName: string;
  creatorImage: string;
  bio: string;
  agentsUploaded: number;
  onClick: () => void;
  index: number;
}

export const CreatorCard: React.FC<CreatorCardProps> = ({
  creatorName,
  creatorImage,
  bio,
  agentsUploaded,
  onClick,
  index,
}) => {
  const backgroundColor = BACKGROUND_COLORS[index % BACKGROUND_COLORS.length];

  return (
    <div
      className={`h-[264px] w-full px-[18px] pb-5 pt-6 ${backgroundColor} inline-flex cursor-pointer flex-col items-start justify-start gap-3.5 rounded-[26px] transition-all duration-200 hover:brightness-95`}
      onClick={onClick}
      data-testid="creator-card"
    >
      <div className="relative h-[64px] w-[64px]">
        <div className="absolute inset-0 overflow-hidden rounded-full">
          <Image
            src={creatorImage}
            alt={creatorName}
            width={64}
            height={64}
            className="h-full w-full object-cover"
            priority
          />
        </div>
      </div>

      <div className="flex flex-1 flex-col items-start justify-start self-stretch">
        <div className="mb-1 self-stretch font-['Poppins'] text-2xl font-semibold leading-loose text-neutral-800">
          {creatorName}
        </div>
        <div className="line-clamp-2 self-stretch font-['Geist'] text-base font-normal leading-normal text-neutral-800">
          {bio}
        </div>
      </div>

      <div className="self-stretch font-['Geist'] text-lg font-semibold leading-7 text-neutral-800">
        {agentsUploaded} agents
      </div>
    </div>
  );
};
