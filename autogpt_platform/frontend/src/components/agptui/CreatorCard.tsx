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
      className={`aspect-square w-80 space-y-4 rounded-3xl bg-amber-100 p-5 pt-6 hover:cursor-pointer hover:bg-amber-200`}
      onClick={onClick}
      data-testid="creator-card"
    >
      {creatorImage ? (
        <Image
          src={creatorImage}
          alt={creatorName}
          width={84}
          height={84}
          className="rounded-full"
          priority
        />
      ) : (
        <div className="h-20 w-20 rounded-full bg-neutral-300 dark:bg-neutral-600" />
      )}

      <div className="flex h-36 flex-col gap-2">
        <h3 className="line-clamp-1 font-poppins text-3xl font-medium text-zinc-800 dark:text-neutral-100">
          {creatorName}
        </h3>
        <p className="line-clamp-3 font-sans text-base font-normal text-zinc-600 dark:text-neutral-400">
          {bio}
        </p>
      </div>

      <div className="font-sans text-sm font-medium text-zinc-800 dark:text-neutral-200">
        {agentsUploaded} agents
      </div>
    </div>
  );
};
