"use client";

import { StarRatingIcons } from "@/components/__legacy__/ui/icons";
import { Separator } from "@/components/__legacy__/ui/separator";
import Link from "next/link";
import { User } from "@supabase/supabase-js";
import { cn } from "@/lib/utils";
import { useAgentInfo } from "./useAgentInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

interface AgentInfoProps {
  user: User | null;
  name: string;
  creator: string;
  shortDescription: string;
  longDescription: string;
  rating: number;
  runs: number;
  categories: string[];
  lastUpdated: string;
  version: string;
  storeListingVersionId: string;
  libraryAgent: LibraryAgent | undefined;
}

export const AgentInfo = ({
  user,
  name,
  creator,
  shortDescription,
  longDescription,
  rating,
  runs,
  categories,
  lastUpdated,
  version,
  storeListingVersionId,
  libraryAgent,
}: AgentInfoProps) => {
  const {
    handleDownload,
    isDownloadingAgent,
    handleLibraryAction,
    isAddingAgentToLibrary,
  } = useAgentInfo({ storeListingVersionId });

  return (
    <div className="w-full max-w-[396px] px-4 sm:px-6 lg:w-[396px] lg:px-0">
      {/* Title */}
      <div
        data-testid="agent-title"
        className="mb-3 w-full font-poppins text-2xl font-medium leading-normal text-neutral-900 dark:text-neutral-100 sm:text-3xl lg:mb-4 lg:text-[35px] lg:leading-10"
      >
        {name}
      </div>

      {/* Creator */}
      <div className="mb-3 flex w-full items-center gap-1.5 lg:mb-4">
        <div className="text-base font-normal text-neutral-800 dark:text-neutral-200 sm:text-lg lg:text-xl">
          by
        </div>
        <Link
          data-testid={"agent-creator"}
          href={`/marketplace/creator/${encodeURIComponent(creator)}`}
          className="text-base font-medium text-neutral-800 hover:underline dark:text-neutral-200 sm:text-lg lg:text-xl"
        >
          {creator}
        </Link>
      </div>

      {/* Short Description */}
      <div className="mb-4 line-clamp-2 w-full text-base font-normal leading-normal text-neutral-600 dark:text-neutral-300 sm:text-lg lg:mb-6 lg:text-xl lg:leading-7">
        {shortDescription}
      </div>

      {/* Buttons */}
      <div className="mb-4 flex w-full gap-3 lg:mb-[60px]">
        {user && (
          <button
            className={cn(
              "inline-flex min-w-24 items-center justify-center rounded-full bg-violet-600 px-4 py-3",
              "transition-colors duration-200 hover:bg-violet-500 disabled:bg-zinc-400",
            )}
            data-testid={"agent-add-library-button"}
            onClick={handleLibraryAction}
            disabled={isAddingAgentToLibrary}
          >
            <span className="justify-start font-sans text-sm font-medium leading-snug text-primary-foreground">
              {libraryAgent ? "See runs" : "Add to library"}
            </span>
          </button>
        )}
        <button
          className={cn(
            "inline-flex min-w-24 items-center justify-center rounded-full bg-zinc-200 px-4 py-3",
            "transition-colors duration-200 hover:bg-zinc-200/70 disabled:bg-zinc-200/40",
          )}
          data-testid={"agent-download-button"}
          onClick={handleDownload}
          disabled={isDownloadingAgent}
        >
          <div className="justify-start text-center font-sans text-sm font-medium leading-snug text-zinc-800">
            Download agent
          </div>
        </button>
      </div>

      {/* Rating and Runs */}
      <div className="mb-4 flex w-full items-center justify-between lg:mb-[44px]">
        <div className="flex items-center gap-1.5 sm:gap-2">
          <span className="whitespace-nowrap text-base font-semibold text-neutral-800 dark:text-neutral-200 sm:text-lg">
            {rating.toFixed(1)}
          </span>
          <div className="flex gap-0.5">{StarRatingIcons(rating)}</div>
        </div>
        <div className="whitespace-nowrap text-base font-semibold text-neutral-800 dark:text-neutral-200 sm:text-lg">
          {runs.toLocaleString()} runs
        </div>
      </div>

      {/* Separator */}
      <Separator className="mb-4 lg:mb-[44px]" />

      {/* Description Section */}
      <div className="mb-4 w-full lg:mb-[36px]">
        <div className="decoration-skip-ink-none mb-1.5 text-base font-medium leading-6 text-neutral-800 dark:text-neutral-200 sm:mb-2">
          Description
        </div>
        <div
          data-testid={"agent-description"}
          className="whitespace-pre-line text-base font-normal leading-6 text-neutral-600 dark:text-neutral-400"
        >
          {longDescription}
        </div>
      </div>

      {/* Categories */}
      <div className="mb-4 flex w-full flex-col gap-1.5 sm:gap-2 lg:mb-[36px]">
        <div className="decoration-skip-ink-none mb-1.5 text-base font-medium leading-6 text-neutral-800 dark:text-neutral-200 sm:mb-2">
          Categories
        </div>
        <div className="flex flex-wrap gap-1.5 sm:gap-2">
          {categories.map((category, index) => (
            <div
              key={index}
              className="decoration-skip-ink-none whitespace-nowrap rounded-full border border-neutral-600 bg-white px-2 py-0.5 text-base font-normal leading-6 text-neutral-800 underline-offset-[from-font] dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-200 sm:px-[16px] sm:py-[10px]"
            >
              {category}
            </div>
          ))}
        </div>
      </div>

      {/* Version History */}
      <div className="flex w-full flex-col gap-0.5 sm:gap-1">
        <div className="decoration-skip-ink-none mb-1.5 text-base font-medium leading-6 text-neutral-800 dark:text-neutral-200 sm:mb-2">
          Version history
        </div>
        <div className="decoration-skip-ink-none text-base font-normal leading-6 text-neutral-600 underline-offset-[from-font] dark:text-neutral-400">
          Last updated {lastUpdated}
        </div>
        <div className="text-xs text-neutral-600 dark:text-neutral-400 sm:text-sm">
          Version {version}
        </div>
      </div>
    </div>
  );
};
