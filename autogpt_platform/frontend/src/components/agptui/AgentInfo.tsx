"use client";

import * as React from "react";
import { IconPlay, IconStar, StarRatingIcons } from "@/components/ui/icons";
import Link from "next/link";
import { Separator } from "@/components/ui/separator";

interface AgentInfoProps {
  name: string;
  creator: string;
  shortDescription: string;
  longDescription: string;
  rating: number;
  runs: number;
  categories: string[];
  lastUpdated: string;
  version: string;
}

export const AgentInfo: React.FC<AgentInfoProps> = ({
  name,
  creator,
  shortDescription,
  longDescription,
  rating,
  runs,
  categories,
  lastUpdated,
  version,
}) => {
  return (
    <div className="w-full max-w-[396px] lg:w-[396px] px-4 sm:px-6 lg:px-0">
      {/* Title */}
      <div className="w-full text-neutral-900 text-2xl sm:text-3xl lg:text-[35px] font-medium font-['Poppins'] leading-normal lg:leading-10 mb-3 lg:mb-4">
        {name}
      </div>

      {/* Creator */}
      <div className="w-full flex items-center gap-1.5 mb-3 lg:mb-4">
        <div className="text-neutral-800 text-base sm:text-lg lg:text-xl font-normal font-['Geist']">
          by
        </div>
        <div className="text-neutral-800 text-base sm:text-lg lg:text-xl font-medium font-['Geist']">
          {creator}
        </div>
      </div>

      {/* Short Description */}
      <div className="w-full text-neutral-600 text-base sm:text-lg lg:text-xl font-normal font-['Geist'] leading-normal lg:leading-7 mb-4 lg:mb-6 line-clamp-2">
        {shortDescription}
      </div>

      {/* Run Agent Button */}
      <div className="w-full mb-4 lg:mb-6">
        <button className="w-full sm:w-auto px-4 sm:px-5 lg:px-6 py-3 sm:py-3.5 lg:py-4 bg-violet-600 hover:bg-violet-700 transition-colors rounded-[38px] inline-flex items-center justify-center gap-2 sm:gap-2.5">
          <IconPlay className="w-5 h-5 sm:w-5 sm:h-5 lg:w-6 lg:h-6 text-white" />
          <span className="text-neutral-50 text-base sm:text-lg font-medium font-['Poppins']">
            Run agent
          </span>
        </button>
      </div>

      {/* Rating and Runs */}
      <div className="w-full flex justify-between items-center mb-4 lg:mb-6">
        <div className="flex items-center gap-1.5 sm:gap-2">
          <span className="text-neutral-800 text-base sm:text-lg font-semibold font-['Geist'] whitespace-nowrap">
            {rating.toFixed(1)}
          </span>
          <div className="flex gap-0.5">
            {StarRatingIcons(rating)}
          </div>
        </div>
        <div className="text-neutral-800 text-base sm:text-lg font-semibold font-['Geist'] whitespace-nowrap">
          {runs.toLocaleString()} runs
        </div>
      </div>

      {/* Separator */}
      <Separator className="mb-4 lg:mb-6" />

      {/* Description Section */}
      <div className="w-full mb-4 lg:mb-6">
        <div className="text-neutral-800 text-xs sm:text-sm font-medium mb-1.5 sm:mb-2">
          Description
        </div>
        <div className="w-full text-neutral-600 text-sm sm:text-base font-normal font-['Geist'] whitespace-pre-line">
          {longDescription}
        </div>
      </div>

      {/* Categories */}
      <div className="w-full flex flex-col gap-1.5 sm:gap-2 mb-4 lg:mb-6">
        <div className="text-neutral-800 text-xs sm:text-sm font-medium">
          Categories
        </div>
        <div className="flex flex-wrap gap-1.5 sm:gap-2">
          {categories.map((category, index) => (
            <div
              key={index}
              className="px-2 sm:px-3 py-0.5 sm:py-1 bg-white rounded-full border border-neutral-200 text-neutral-800 text-xs sm:text-sm whitespace-nowrap"
            >
              {category}
            </div>
          ))}
        </div>
      </div>

      {/* Rate Agent */}
      <div className="w-full flex flex-col gap-1.5 sm:gap-2 mb-4 lg:mb-6">
        <div className="text-neutral-800 text-xs sm:text-sm font-medium">
          Rate agent
        </div>
        <div className="flex gap-1">
          {[1, 2, 3, 4, 5].map((star) => (
            <IconStar 
              key={star} 
              className="w-4 h-4 sm:w-5 sm:h-5 text-neutral-300 cursor-pointer hover:text-neutral-800"
            />
          ))}
        </div>
      </div>

      {/* Version History */}
      <div className="w-full flex flex-col gap-0.5 sm:gap-1">
        <div className="text-neutral-800 text-xs sm:text-sm font-medium">
          Version history
        </div>
        <div className="text-neutral-600 text-xs sm:text-sm">
          Last updated {lastUpdated}
        </div>
        <div className="text-neutral-600 text-xs sm:text-sm">
          Version {version}
        </div>
      </div>
    </div>
  );
};
