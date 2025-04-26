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
      className={`h-64 w-full px-5 pb-5 pt-6 md:w-80 ${backgroundColor} inline-flex cursor-pointer flex-col items-start justify-start gap-3.5 rounded-[1.5rem] transition-all duration-200 hover:brightness-95`}
      onClick={onClick}
      data-testid="creator-card"
    >
      {creatorImage ? (
        <Image
          src={creatorImage}
          alt={creatorName}
          width={80}
          height={80}
          className="h-20 w-20 rounded-full"
          priority
        />
      ) : (
        <div className="h-20 w-20 rounded-full bg-neutral-300 dark:bg-neutral-600" />
      )}

      <div className="flex flex-col gap-2">
        <h3 className="line-clamp-1 font-poppins text-2xl font-semibold text-neutral-800 dark:text-neutral-100">
          {creatorName}
        </h3>
        <p className="line-clamp-2 font-sans text-base font-normal text-neutral-800 dark:text-neutral-400">
          {bio}
        </p>
        <div className="font-sans text-lg font-semibold text-neutral-800 dark:text-neutral-200">
          {agentsUploaded} agents
        </div>
      </div>
    </div>
  );
};
