"use client";

import { StarRatingIcons } from "@/components/__legacy__/ui/icons";
import { Separator } from "@/components/__legacy__/ui/separator";
import Link from "next/link";
import { User } from "@supabase/supabase-js";
import { cn } from "@/lib/utils";
import { useAgentInfo } from "./useAgentInfo";
import { useGetV2GetSpecificAgent } from "@/app/api/__generated__/endpoints/store/store";
import { Text } from "@/components/atoms/Text/Text";

interface AgentInfoProps {
  user: User | null;
  agentId: string;
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
  isAgentAddedToLibrary: boolean;
  creatorSlug?: string;
  agentSlug?: string;
}

export const AgentInfo = ({
  user,
  agentId,
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
  isAgentAddedToLibrary,
  creatorSlug,
  agentSlug,
}: AgentInfoProps) => {
  const {
    handleDownload,
    isDownloadingAgent,
    handleLibraryAction,
    isAddingAgentToLibrary,
  } = useAgentInfo({ storeListingVersionId });

  // Get store agent data for version history
  const { data: storeAgentData } = useGetV2GetSpecificAgent(
    creatorSlug || "",
    agentSlug || "",
    {
      query: {
        enabled: !!(creatorSlug && agentSlug),
      },
    },
  );

  // Process version data for display - limit to latest 4 versions
  const agentVersions =
    storeAgentData?.status === 200 && storeAgentData.data.agentGraphVersions
      ? storeAgentData.data.agentGraphVersions
          .map((versionStr: string) => parseInt(versionStr, 10))
          .sort((a: number, b: number) => b - a)
          .slice(0, 4) // Only show latest 4 versions
          .map((versionNum: number) => ({
            version: versionNum,
            isCurrentVersion: versionNum.toString() === version,
          }))
      : [];

  // Generate sample change descriptions for versions
  const getVersionDescription = (version: number): string => {
    const descriptions = [
      "Performance improvements and bug fixes",
      "Enhanced error handling and stability",
      "New features and UI improvements",
      "Security updates and optimizations",
      "Code refactoring and minor fixes",
      "Updated dependencies and compatibility",
    ];
    return descriptions[version % descriptions.length];
  };

  const renderVersionItem = (versionInfo: {
    version: number;
    isCurrentVersion: boolean;
  }) => {
    return (
      <div key={versionInfo.version} className="py-1">
        <div className="flex items-center gap-2">
          <Text
            variant="small"
            className={`font-semibold ${versionInfo.isCurrentVersion ? "text-blue-600 dark:text-blue-400" : "text-neutral-700 dark:text-neutral-300"}`}
          >
            v{versionInfo.version}
          </Text>
          {versionInfo.isCurrentVersion && (
            <span className="rounded bg-blue-100 px-1.5 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-900 dark:text-blue-100">
              Current
            </span>
          )}
          <Text
            variant="small"
            className="text-neutral-600 dark:text-neutral-400"
          >
            - {getVersionDescription(versionInfo.version)}
          </Text>
        </div>
      </div>
    );
  };

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
      <div className="mb-4 line-clamp-2 w-full text-base font-normal leading-normal text-neutral-600 dark:text-neutral-300 sm:text-lg lg:mb-5 lg:text-xl lg:leading-7">
        {shortDescription}
      </div>

      {/* Rating and Runs */}
      <div className="flex w-full items-center justify-between">
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

      {/* Buttons */}
      {user && (
        <div className="mt-6 flex w-full gap-3 lg:mt-8">
          <button
            className={cn(
              "inline-flex min-w-24 items-center justify-center rounded-full bg-violet-600 px-4 py-3",
              "transition-colors duration-200 hover:bg-violet-500 disabled:bg-zinc-400",
            )}
            data-testid={"agent-add-library-button"}
            disabled={isAddingAgentToLibrary}
            onClick={() =>
              handleLibraryAction({
                isAddingAgentFirstTime: !isAgentAddedToLibrary,
              })
            }
          >
            <span className="justify-start font-sans text-sm font-medium leading-snug text-primary-foreground">
              {isAgentAddedToLibrary ? "See runs" : "Add to library"}
            </span>
          </button>
        </div>
      )}

      {/* Download section */}
      <p className="mt-6 text-zinc-600 dark:text-zinc-400 lg:mt-12">
        Want to use this agent locally?{" "}
        <button
          className="underline"
          onClick={() => handleDownload(agentId, name)}
          disabled={isDownloadingAgent}
          data-testid="agent-download-button"
        >
          Download here.
        </button>
      </p>

      {/* Separator */}
      <Separator className="my-7" />

      {/* Agent Details Section */}
      <div className="flex w-full flex-col gap-4 lg:gap-6">
        {/* Description Section */}
        <div className="w-full">
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
        <div className="flex w-full flex-col gap-1.5 sm:gap-2">
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
        <div className="flex w-full flex-col gap-1.5 sm:gap-2">
          <div className="decoration-skip-ink-none mb-1.5 text-base font-medium leading-6 text-neutral-800 dark:text-neutral-200 sm:mb-2">
            Version history
          </div>
          <div className="decoration-skip-ink-none text-base font-normal leading-6 text-neutral-600 underline-offset-[from-font] dark:text-neutral-400">
            Last updated {lastUpdated}
          </div>

          {/* Version List */}
          {agentVersions.length > 0 ? (
            <div className="mt-2">
              {agentVersions.map(renderVersionItem)}
              {storeAgentData?.status === 200 &&
                storeAgentData.data.agentGraphVersions &&
                storeAgentData.data.agentGraphVersions.length > 4 && (
                  <div className="py-1 text-xs text-neutral-500 dark:text-neutral-400">
                    ... and {storeAgentData.data.agentGraphVersions.length - 4}{" "}
                    more versions
                  </div>
                )}
            </div>
          ) : (
            <div className="text-xs text-neutral-600 dark:text-neutral-400 sm:text-sm">
              Version {version}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
