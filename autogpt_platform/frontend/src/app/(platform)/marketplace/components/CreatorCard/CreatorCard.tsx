import Image from "next/image";
import { backgroundColor } from "./helper";

interface CreatorCardProps {
  creatorName: string;
  creatorImage: string | null;
  bio: string;
  agentsUploaded: number;
  onClick: () => void;
  index: number;
}

export const CreatorCard = ({
  creatorName,
  creatorImage,
  bio,
  agentsUploaded,
  onClick,
  index,
}: CreatorCardProps) => {
  return (
    <div
      className={`h-[264px] w-full px-[18px] pt-6 pb-5 ${backgroundColor(index)} inline-flex cursor-pointer flex-col items-start justify-start gap-3.5 rounded-[26px] transition-all duration-200 hover:brightness-95`}
      onClick={onClick}
      data-testid="creator-card"
    >
      <div className="relative h-[64px] w-[64px]">
        <div className="absolute inset-0 overflow-hidden rounded-full">
          {creatorImage ? (
            <Image
              src={creatorImage}
              alt={creatorName}
              width={64}
              height={64}
              className="h-full w-full object-cover"
            />
          ) : (
            <div className="h-full w-full bg-neutral-300 dark:bg-neutral-600" />
          )}
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <h3 className="font-poppins text-2xl leading-tight font-semibold text-neutral-900 dark:text-neutral-100">
          {creatorName}
        </h3>
        <p className="text-sm leading-normal font-normal text-neutral-600 dark:text-neutral-400">
          {bio}
        </p>
        <div className="text-lg leading-7 font-semibold text-neutral-800 dark:text-neutral-200">
          {agentsUploaded} agents
        </div>
      </div>
    </div>
  );
};
