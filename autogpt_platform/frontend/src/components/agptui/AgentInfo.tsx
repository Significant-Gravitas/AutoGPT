import * as React from "react";
import { Button } from "./Button";
import { StarIcon, StarFilledIcon } from "@radix-ui/react-icons";
import Link from "next/link";

interface AgentInfoProps {
  onRunAgent: () => void;
  name: string;
  creator: string;
  description: string;
  rating: number;
  runs: number;
  categories: string[];
  lastUpdated: string;
  version: string;
}

export const AgentInfo: React.FC<AgentInfoProps> = ({
  onRunAgent,
  name,
  creator,
  description,
  rating,
  runs,
  categories,
  lastUpdated,
  version,
}) => {
  const renderStars = (rating: number) => {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 !== 0;
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
    <div className="flow-root w-[27.5rem]">
      <div className="font-['PP Neue Montreal TT'] mb-4 text-5xl font-medium leading-9 tracking-wide text-[#272727]">
        {name}
      </div>
      <div className="font-['PP Neue Montreal TT'] mb-4 text-[1.1875rem] font-medium leading-9 tracking-tight text-[#878787]">
        by{" "}
        <Link
          href={`/creator/${creator.replace(/\s+/g, "-")}`}
          className="text-[#272727]"
        >
          {creator}
        </Link>
      </div>
      <Button onClick={onRunAgent} className="mb-8" variant="outline">
        Run agent
      </Button>
      <div className="font-['PP Neue Montreal TT'] mb-6 text-[1.1875rem] font-normal leading-relaxed tracking-tight text-[#282828]">
        {description}
      </div>
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center">
          <div className="font-['PP Neue Montreal TT'] mr-2 text-xl font-normal tracking-tight text-[#272727]">
            {rating.toFixed(1)}
          </div>
          <div className="flex items-center gap-px">{renderStars(rating)}</div>
        </div>
        <div>
          <span className="font-['PP Neue Montreal TT'] text-xl font-medium tracking-tight text-[#272727]">
            {runs.toLocaleString()}+
          </span>
          <span className="font-['PP Neue Montreal TT'] text-xl font-normal tracking-tight text-[#272727]">
            {" "}
            runs
          </span>
        </div>
      </div>
      <div className="font-['PP Neue Montreal TT'] mb-3 text-lg font-medium leading-9 tracking-tight text-[#282828]">
        Categories
      </div>
      <div className="mb-6 flex flex-wrap gap-2.5">
        {categories.map((category, index) => (
          <div
            key={index}
            className="flex items-center rounded-[2.125rem] border border-black/50 px-4 py-1.5"
          >
            <div className="font-['PP Neue Montreal TT'] text-[1.1875rem] font-normal leading-relaxed tracking-tight text-[#474747]">
              {category}
            </div>
          </div>
        ))}
      </div>
      <div className="font-['PP Neue Montreal TT'] mb-3 text-lg font-medium leading-9 tracking-tight text-[#282828]">
        Version history
      </div>
      <div className="font-['PP Neue Montreal TT'] mb-2 text-[1.1875rem] font-normal leading-relaxed tracking-tight text-[#474747]">
        Last updated {lastUpdated}
      </div>
      <div className="font-['PP Neue Montreal TT'] text-[1.1875rem] font-normal leading-relaxed tracking-tight text-[#474747]">
        Version {version}
      </div>
    </div>
  );
};
