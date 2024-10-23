import * as React from "react";
import { Button } from "./Button";
import Link from "next/link";
import { StarRatingIcons } from "@/components/ui/icons";
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
  return (
    <div className="flow-root w-full lg:w-[27.5rem]">
      <div className="mb-2 font-neue text-3xl font-medium tracking-wide text-[#272727] md:mb-4 md:text-4xl lg:text-5xl">
        {name}
      </div>
      <div className="mb-2 font-neue text-lg font-medium leading-9 tracking-tight text-[#737373] md:mb-4 md:text-xl lg:text-2xl">
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
      <div className="mb-6 mr-6 flex items-center justify-between">
        <div className="flex items-center">
          <div className="font-['PP Neue Montreal TT'] mr-2 text-xl font-normal tracking-tight text-[#272727]">
            {rating.toFixed(1)}
          </div>
          <div className="flex items-center gap-px">
            {StarRatingIcons(rating)}
          </div>
        </div>
        <div>
          <span className="font-neue text-xl font-medium tracking-tight text-[#272727]">
            {runs.toLocaleString()}+
          </span>
          <span className="font-neue text-xl font-normal tracking-tight text-[#272727]">
            {" "}
            runs
          </span>
        </div>
      </div>
      <div className="mb-3 font-neue text-lg font-medium leading-9 tracking-tight text-[#282828]">
        Categories
      </div>
      <div className="mb-6 flex flex-wrap gap-2.5">
        {categories.map((category, index) => (
          <div
            key={index}
            className="flex items-center rounded-[2.125rem] border border-black/50 px-4 py-1.5"
          >
            <div className="font-neue text-[1.1875rem] font-normal leading-relaxed tracking-tight text-[#474747]">
              {category}
            </div>
          </div>
        ))}
      </div>
      <div className="mb-3 font-neue text-lg font-medium leading-9 tracking-tight text-[#282828]">
        Version history
      </div>
      <div className="mb-2 font-neue text-[1.1875rem] font-normal leading-relaxed tracking-tight text-[#474747]">
        Last updated {lastUpdated}
      </div>
      <div className="font-neue text-[1.1875rem] font-normal leading-relaxed tracking-tight text-[#474747]">
        Version {version}
      </div>
    </div>
  );
};
