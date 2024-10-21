import * as React from "react";
import { StarIcon, StarFilledIcon } from "@radix-ui/react-icons";

interface CreatorDetailsProps {
  name: string;
  username: string;
  description: string;
  avgRating: number;
  agentCount: number;
  topCategories: string[];
  otherLinks: {
    website?: string;
    github?: string;
    linkedin?: string;
  };
}

export const CreatorDetails: React.FC<CreatorDetailsProps> = ({
  name,
  username,
  description,
  avgRating,
  agentCount,
  topCategories,
  otherLinks,
}) => {
  const renderStars = () => {
    const fullStars = Math.floor(avgRating);
    const hasHalfStar = avgRating % 1 !== 0;
    const stars = [];

    for (let i = 0; i < 5; i++) {
      if (i < fullStars) {
        stars.push(<StarFilledIcon key={i} className="text-black" />);
      } else if (i === fullStars && hasHalfStar) {
        stars.push(<StarIcon key={i} className="text-black" />);
      } else {
        stars.push(<StarIcon key={i} className="text-black" />);
      }
    }

    return stars;
  };

  return (
    <div className="relative w-full max-w-[1359px]">
      <div className="flex flex-col gap-8 lg:flex-row">
        <div className="w-full lg:w-[402px]">
          <div className="mb-8">
            <h1 className="font-['PP Neue Montreal TT'] mb-2 text-5xl font-medium leading-9 tracking-wide text-[#272727]">
              {name}
            </h1>
            <p className="font-['PP Neue Montreal TT'] text-[19px] font-medium leading-9 tracking-tight text-[#878787]">
              @{username}
            </p>
          </div>
          <div className="mb-8 flex h-[127px] w-[127px] items-center justify-center rounded-full bg-[#d9d9d9]">
            <span className="font-['SF Pro'] text-center text-[64px] font-normal text-[#7e7e7e]">
              􀉪
            </span>
          </div>
          <div>
            <h2 className="font-['PP Neue Montreal TT'] mb-2 text-lg font-medium leading-9 tracking-tight text-[#282828]">
              Other links
            </h2>
            <div className="flex gap-9">
              {otherLinks.website && (
                <a
                  href={otherLinks.website}
                  className="flex items-center gap-3"
                >
                  <span className="h-6 w-6 opacity-80">🌐</span>
                  <span className="font-['PP Neue Montreal TT'] text-lg font-normal tracking-tight text-[#282828]">
                    Website
                  </span>
                </a>
              )}
              {otherLinks.github && (
                <a href={otherLinks.github} className="flex items-center gap-3">
                  <span className="h-6 w-6 opacity-80">🐙</span>
                  <span className="font-['PP Neue Montreal TT'] text-lg font-normal tracking-tight text-[#282828]">
                    GitHub
                  </span>
                </a>
              )}
              {otherLinks.linkedin && (
                <a
                  href={otherLinks.linkedin}
                  className="flex items-center gap-3"
                >
                  <span className="h-6 w-6 opacity-80">🔗</span>
                  <span className="font-['PP Neue Montreal TT'] text-lg font-normal tracking-tight text-[#282828]">
                    LinkedIn
                  </span>
                </a>
              )}
            </div>
          </div>
        </div>
        <div className="w-full lg:w-[669px]">
          <p className="font-['PP Neue Montreal TT'] mb-8 text-3xl font-normal leading-[39px] tracking-tight text-[#474747]">
            {description}
          </p>
          <div className="mb-8 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="font-['PP Neue Montreal TT'] text-xl font-normal tracking-tight text-[#474747]">
                {avgRating.toFixed(1)}
              </span>
              <div className="flex">{renderStars()}</div>
              <span className="font-['PP Neue Montreal TT'] text-xl font-medium tracking-tight text-[#474747]">
                avg rating
              </span>
            </div>
            <span className="font-['PP Neue Montreal TT'] text-xl font-medium tracking-tight text-[#474747]">
              {agentCount} agents
            </span>
          </div>
          <div>
            <h2 className="font-['PP Neue Montreal TT'] mb-4 text-lg font-medium leading-9 tracking-tight text-[#282828]">
              Top categories
            </h2>
            <div className="flex flex-wrap gap-2.5">
              {topCategories.map((category, index) => (
                <div
                  key={index}
                  className="rounded-[34px] border border-black/50 px-4 py-1.5"
                >
                  <span className="font-['PP Neue Montreal TT'] text-[19px] font-normal leading-relaxed tracking-tight text-[#474747]">
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
};
