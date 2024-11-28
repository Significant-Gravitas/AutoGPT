import * as React from "react";
import Image from "next/image";

const BACKGROUND_COLORS = [
  "bg-amber-100 dark:bg-amber-800", // #fef3c7 / #92400e
  "bg-violet-100 dark:bg-violet-800", // #ede9fe / #5b21b6
  "bg-green-100 dark:bg-green-800", // #dcfce7 / #065f46
  "bg-blue-100 dark:bg-blue-800", // #dbeafe / #1e3a8a
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
        <div className="mb-1 self-stretch font-['Poppins'] text-2xl font-semibold leading-loose text-neutral-800 dark:text-neutral-200">
          {creatorName}
        </div>
        <div className="line-clamp-2 self-stretch font-['Geist'] text-base font-normal leading-normal text-neutral-800 dark:text-neutral-200">
          {bio}
        </div>
      </div>

      <div className="self-stretch font-['Geist'] text-lg font-semibold leading-7 text-neutral-800 dark:text-neutral-200">
        {agentsUploaded} agents
      </div>
    </div>
  );
};
