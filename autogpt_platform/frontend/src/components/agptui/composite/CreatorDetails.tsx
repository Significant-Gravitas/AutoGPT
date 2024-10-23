import * as React from "react";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { getIconForSocial, StarRatingIcons } from "@/components/ui/icons";

interface CreatorDetailsProps {
  name: string;
  username: string;
  description: string;
  avgRating: number;
  agentCount: number;
  topCategories: string[];
  otherLinks: Record<string, string>;
  avatarSrc: string;
}

export const CreatorDetails: React.FC<CreatorDetailsProps> = React.memo(
  ({
    name,
    username,
    description,
    avgRating,
    agentCount,
    topCategories,
    otherLinks,
    avatarSrc,
  }) => {
    const avatarSizeClasses = "h-32 w-32";

    return (
      <div className="w-full max-w-[1359px]">
        {/* Left Hand Side */}
        <div className="flex flex-col gap-8 lg:flex-row">
          <div className="w-full lg:w-1/2">
            <div className="mb-8">
              <h1 className="mb-2 font-neue text-5xl font-medium leading-9 tracking-wide text-[#272727]">
                {name}
              </h1>
              <p className="font-neue text-[19px] font-medium leading-9 tracking-tight text-[#737373]">
                @{username}
              </p>
            </div>
            <div className="mb-8">
              <Avatar className={avatarSizeClasses}>
                <AvatarImage src={avatarSrc} alt={`${name}'s avatar`} />
                <AvatarFallback className={avatarSizeClasses}>
                  {username.charAt(0)}
                </AvatarFallback>
              </Avatar>
            </div>
            {Object.keys(otherLinks).length > 0 && (
              <div>
                <h2 className="mb-2 font-neue text-lg font-medium leading-9 tracking-tight text-[#282828]">
                  Other links
                </h2>
                <div className="flex flex-wrap gap-4">
                  {Object.entries(otherLinks)
                    .slice(0, 5)
                    .map(([key, url]) => (
                      <a
                        key={key}
                        href={url}
                        className="flex items-center gap-2"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {getIconForSocial(url, { className: "h-6 w-6" })}
                        <span className="font-neue text-lg font-normal tracking-tight text-[#282828]">
                          {key}
                        </span>
                      </a>
                    ))}
                </div>
              </div>
            )}
          </div>
          {/* Right Hand Side */}
          <div className="w-full lg:w-1/2">
            <p className="mb-8 font-neue text-3xl font-normal leading-[39px] tracking-tight text-[#474747]">
              {description || "This creator is keeping their bio under wraps"}
            </p>
            <div className="mb-8 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-neue text-xl font-normal tracking-tight text-[#474747]">
                  {avgRating.toFixed(1)}
                </span>
                <div className="flex">{StarRatingIcons(avgRating)}</div>
                <span className="font-neue text-xl font-medium tracking-tight text-[#474747]">
                  avg rating
                </span>
              </div>
              <span className="font-neue text-xl font-medium tracking-tight text-[#474747]">
                {agentCount} agents
              </span>
            </div>
            <div>
              <h2 className="mb-4 font-neue text-lg font-medium leading-9 tracking-tight text-[#282828]">
                Top categories
              </h2>
              <div className="flex flex-wrap gap-2.5">
                {topCategories.map((category, index) => (
                  <div
                    key={index}
                    className="rounded-[34px] border border-black/50 px-4 py-1.5"
                  >
                    <span className="font-neue text-[19px] font-normal leading-relaxed tracking-tight text-[#474747]">
                      {category}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  },
);

CreatorDetails.displayName = "CreatorDetails";
