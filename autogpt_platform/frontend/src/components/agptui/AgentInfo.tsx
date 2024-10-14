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
    <div className="w-[27.5rem] flow-root">
      <div className="text-[#272727] text-5xl font-medium font-['PP Neue Montreal TT'] leading-9 tracking-wide mb-4">
        {name}
      </div>
      <div className="text-[#878787] text-[1.1875rem] font-medium font-['PP Neue Montreal TT'] leading-9 tracking-tight mb-4">
        by <Link href={`/creator/${creator.replace(/\s+/g, '-')}`} className="text-[#272727]">{creator}</Link>
      </div>
      <Button
        onClick={onRunAgent}
        className="mb-8"
        variant="outline"
      >
        Run agent
      </Button>
      <div className="text-[#282828] text-[1.1875rem] font-normal font-['PP Neue Montreal TT'] leading-relaxed tracking-tight mb-6">
        {description}
      </div>
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center">
          <div className="text-[#272727] text-xl font-normal font-['PP Neue Montreal TT'] tracking-tight mr-2">
            {rating.toFixed(1)}
          </div>
          <div className="flex items-center gap-px">
            {renderStars(rating)}
          </div>
        </div>
        <div>
          <span className="text-[#272727] text-xl font-medium font-['PP Neue Montreal TT'] tracking-tight">
            {runs.toLocaleString()}+
          </span>
          <span className="text-[#272727] text-xl font-normal font-['PP Neue Montreal TT'] tracking-tight">
            {" "}
            runs
          </span>
        </div>
      </div>
      <div className="text-[#282828] text-lg font-medium font-['PP Neue Montreal TT'] leading-9 tracking-tight mb-3">
        Categories
      </div>
      <div className="flex flex-wrap gap-2.5 mb-6">
        {categories.map((category, index) => (
          <div
            key={index}
            className="px-4 py-1.5 rounded-[2.125rem] border border-black/50 flex items-center"
          >
            <div className="text-[#474747] text-[1.1875rem] font-normal font-['PP Neue Montreal TT'] leading-relaxed tracking-tight">
              {category}
            </div>
          </div>
        ))}
      </div>
      <div className="text-[#282828] text-lg font-medium font-['PP Neue Montreal TT'] leading-9 tracking-tight mb-3">
        Version history
      </div>
      <div className="text-[#474747] text-[1.1875rem] font-normal font-['PP Neue Montreal TT'] leading-relaxed tracking-tight mb-2">
        Last updated {lastUpdated}
      </div>
      <div className="text-[#474747] text-[1.1875rem] font-normal font-['PP Neue Montreal TT'] leading-relaxed tracking-tight">
        Version {version}
      </div>
    </div>
  );
};
