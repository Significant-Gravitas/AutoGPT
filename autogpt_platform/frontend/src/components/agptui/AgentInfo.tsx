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
    <div className="w-full max-w-[396px] px-4 sm:px-6 lg:w-[396px] lg:px-0">
      {/* Title */}
      <div className="font-poppins mb-3 w-full text-2xl font-medium leading-normal text-neutral-900 dark:text-neutral-100 sm:text-3xl lg:mb-4 lg:text-[35px] lg:leading-10">
        {name}
      </div>

      {/* Creator */}
      <div className="mb-3 flex w-full items-center gap-1.5 lg:mb-4">
        <div className="font-geist text-base font-normal text-neutral-800 dark:text-neutral-200 sm:text-lg lg:text-xl">
          by
        </div>
        <div className="font-geist text-base font-medium text-neutral-800 dark:text-neutral-200 sm:text-lg lg:text-xl">
          {creator}
        </div>
      </div>

      {/* Short Description */}
      <div className="font-geist mb-4 line-clamp-2 w-full text-base font-normal leading-normal text-neutral-600 dark:text-neutral-300 sm:text-lg lg:mb-6 lg:text-xl lg:leading-7">
        {shortDescription}
      </div>

      {/* Run Agent Button */}
      <div className="mb-4 w-full lg:mb-6">
        <button className="inline-flex w-full items-center justify-center gap-2 rounded-[38px] bg-violet-600 px-4 py-3 transition-colors hover:bg-violet-700 sm:w-auto sm:gap-2.5 sm:px-5 sm:py-3.5 lg:px-6 lg:py-4">
          <IconPlay className="h-5 w-5 text-white sm:h-5 sm:w-5 lg:h-6 lg:w-6" />
          <span className="font-poppins text-base font-medium text-neutral-50 sm:text-lg">
            Run agent
          </span>
        </button>
      </div>

      {/* Rating and Runs */}
      <div className="mb-4 flex w-full items-center justify-between lg:mb-6">
        <div className="flex items-center gap-1.5 sm:gap-2">
          <span className="font-geist whitespace-nowrap text-base font-semibold text-neutral-800 dark:text-neutral-200 sm:text-lg">
            {rating.toFixed(1)}
          </span>
          <div className="flex gap-0.5">{StarRatingIcons(rating)}</div>
        </div>
        <div className="font-geist whitespace-nowrap text-base font-semibold text-neutral-800 dark:text-neutral-200 sm:text-lg">
          {runs.toLocaleString()} runs
        </div>
      </div>

      {/* Separator */}
      <Separator className="mb-4 lg:mb-6" />

      {/* Description Section */}
      <div className="mb-4 w-full lg:mb-6">
        <div className="mb-1.5 text-xs font-medium text-neutral-800 dark:text-neutral-200 sm:mb-2 sm:text-sm">
          Description
        </div>
        <div className="font-geist w-full whitespace-pre-line text-sm font-normal text-neutral-600 dark:text-neutral-300 sm:text-base">
          {longDescription}
        </div>
      </div>

      {/* Categories */}
      <div className="mb-4 flex w-full flex-col gap-1.5 sm:gap-2 lg:mb-6">
        <div className="text-xs font-medium text-neutral-800 dark:text-neutral-200 sm:text-sm">
          Categories
        </div>
        <div className="flex flex-wrap gap-1.5 sm:gap-2">
          {categories.map((category, index) => (
            <div
              key={index}
              className="whitespace-nowrap rounded-full border border-neutral-200 bg-white px-2 py-0.5 text-xs text-neutral-800 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-200 sm:px-3 sm:py-1 sm:text-sm"
            >
              {category}
            </div>
          ))}
        </div>
      </div>

      {/* Version History */}
      <div className="flex w-full flex-col gap-0.5 sm:gap-1">
        <div className="text-xs font-medium text-neutral-800 dark:text-neutral-200 sm:text-sm">
          Version history
        </div>
        <div className="text-xs text-neutral-600 dark:text-neutral-400 sm:text-sm">
          Last updated {lastUpdated}
        </div>
        <div className="text-xs text-neutral-600 dark:text-neutral-400 sm:text-sm">
          Version {version}
        </div>
      </div>
    </div>
  );
};
