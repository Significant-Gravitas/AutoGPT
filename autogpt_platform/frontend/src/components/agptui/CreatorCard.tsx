import * as React from "react";
import Image from "next/image";

const BACKGROUND_COLORS = [
  'bg-amber-100',  // #fef3c7
  'bg-violet-100', // #ede9fe
  'bg-green-100',  // #dcfce7
  'bg-blue-100',   // #dbeafe
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
      className={`w-full h-[264px] px-[18px] pt-6 pb-5 ${backgroundColor} rounded-[26px] flex-col justify-start items-start gap-3.5 inline-flex cursor-pointer hover:brightness-95 transition-all duration-200`}
      onClick={onClick}
      data-testid="creator-card"
    >
      <div className="relative w-[64px] h-[64px]">
        <div className="absolute inset-0 rounded-full overflow-hidden">
          <Image
            src={creatorImage}
            alt={creatorName}
            width={64}
            height={64}
            className="object-cover w-full h-full"
            priority
          />
        </div>
      </div>

      <div className="self-stretch flex-1 flex-col justify-start items-start flex">
        <div className="self-stretch text-neutral-800 text-2xl font-semibold font-['Poppins'] leading-loose mb-1">
          {creatorName}
        </div>
        <div className="self-stretch text-neutral-800 text-base font-normal font-['Geist'] leading-normal line-clamp-2">
          {bio}
        </div>
      </div>

      <div className="self-stretch text-neutral-800 text-lg font-semibold font-['Geist'] leading-7">
        {agentsUploaded} agents
      </div>
    </div>
  );
};
