import Image from "next/image";
import { StarRatingIcons } from "@/components/ui/icons";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";

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
      className="flex h-[27rem] w-full max-w-md cursor-pointer flex-col items-start rounded-3xl bg-background transition-all duration-300 hover:shadow-lg dark:hover:shadow-gray-700"
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
      {/* First Section: Image with Avatar */}
      <div className="relative aspect-[2/1.2] w-full overflow-hidden rounded-3xl md:aspect-[2.17/1]">
        {agentImage && (
          <Image
            src={agentImage}
            alt={`${agentName} preview image`}
            fill
            className="object-cover"
          />
        )}
        {!hideAvatar && (
          <div className="absolute bottom-4 left-4">
            <Avatar className="h-16 w-16">
              {avatarSrc && (
                <AvatarImage
                  src={avatarSrc}
                  alt={`${creatorName || agentName} creator avatar`}
                />
              )}
              <AvatarFallback size={64}>
                {(creatorName || agentName).charAt(0)}
              </AvatarFallback>
            </Avatar>
          </div>
        )}
      </div>

      <div className="mt-3 flex w-full flex-1 flex-col px-4">
        {/* Second Section: Agent Name and Creator Name */}
        <div className="flex w-full flex-col">
          <h3 className="line-clamp-2 font-poppins text-2xl font-semibold text-[#272727] dark:text-neutral-100">
            {agentName}
          </h3>
          {!hideAvatar && creatorName && (
            <p className="mt-3 truncate font-sans text-xl font-normal text-neutral-600 dark:text-neutral-400">
              by {creatorName}
            </p>
          )}
        </div>

        {/* Third Section: Description */}
        <div className="mt-2.5 flex w-full flex-col">
          <p className="line-clamp-3 text-base font-normal leading-normal text-neutral-600 dark:text-neutral-400">
            {description}
          </p>
        </div>

        <div className="flex-grow" />
        {/* Spacer to push stats to bottom */}

        {/* Fourth Section: Stats Row - aligned to bottom */}
        <div className="mt-5 w-full">
          <div className="flex items-center justify-between">
            <div className="text-lg font-semibold text-neutral-800 dark:text-neutral-200">
              {runs.toLocaleString()} runs
            </div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-semibold text-neutral-800 dark:text-neutral-200">
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
    </div>
  );
};
