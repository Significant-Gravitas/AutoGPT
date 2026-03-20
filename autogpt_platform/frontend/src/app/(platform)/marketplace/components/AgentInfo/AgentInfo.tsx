"use client";

import { useGetV2GetSpecificAgent } from "@/app/api/__generated__/endpoints/store/store";
import type { ChangelogEntry } from "@/app/api/__generated__/models/changelogEntry";
import type { GetV2GetSpecificAgentParams } from "@/app/api/__generated__/models/getV2GetSpecificAgentParams";
import type { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { okData } from "@/app/api/helpers";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { formatTimeAgo } from "@/lib/utils/time";
import Link from "next/link";
import { FileArrowDownIcon, PlusIcon } from "@phosphor-icons/react";
import { User } from "@supabase/supabase-js";
import { useAgentInfo } from "./useAgentInfo";

interface AgentInfoProps {
  user: User | null;
  agentId: string;
  name: string;
  creator: string;
  creatorAvatar?: string;
  shortDescription: string;
  longDescription: string;
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
  creatorAvatar,
  shortDescription,
  longDescription,
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
  const params: GetV2GetSpecificAgentParams = { include_changelog: true };
  const { data: storeAgentData } = useGetV2GetSpecificAgent(
    creatorSlug || "",
    agentSlug || "",
    params,
    {
      query: {
        enabled: !!(creatorSlug && agentSlug),
      },
    },
  );

  // Calculate update information using simple helper functions
  const storeData = okData(storeAgentData) as StoreAgentDetails | undefined;

  // Process version data for display - use store listing versions (not agentGraphVersions)
  const allVersions = storeData?.versions
    ? storeData.versions
        .map((versionStr: string) => parseInt(versionStr, 10))
        .sort((a: number, b: number) => b - a)
        .map((versionNum: number) => ({
          version: versionNum,
          isCurrentVersion: false, // We'll update this logic if needed
        }))
    : [];

  const renderVersionItem = (versionInfo: {
    version: number;
    isCurrentVersion: boolean;
  }) => {
    // Find real changelog data for this version
    const changelogEntry = storeData?.changelog?.find(
      (entry: ChangelogEntry) =>
        entry.version === versionInfo.version.toString(),
    );

    return (
      <div key={versionInfo.version} className="mb-6 last:mb-0">
        {/* Version Header */}
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Text
              variant="body"
              className="font-semibold text-neutral-900 dark:text-neutral-100"
            >
              Version {versionInfo.version}.0
            </Text>
            {versionInfo.isCurrentVersion && (
              <span className="rounded bg-blue-100 px-1.5 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-900 dark:text-blue-100">
                Current
              </span>
            )}
          </div>
          {changelogEntry && (
            <Text
              variant="small"
              className="text-neutral-500 dark:text-neutral-400"
            >
              {new Date(changelogEntry.date).toLocaleDateString("en-US", {
                year: "numeric",
                month: "long",
                day: "numeric",
              })}
            </Text>
          )}
        </div>

        {/* Real Changelog Content */}
        {changelogEntry && (
          <div className="space-y-2">
            <Text
              variant="body"
              className="text-neutral-700 dark:text-neutral-300"
            >
              {changelogEntry.changes_summary}
            </Text>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="w-full px-4 sm:px-6 lg:px-0">
      <div className="mb-8 rounded-2xl bg-gradient-to-r from-blue-200 to-indigo-200 p-[1px]">
        <div className="flex flex-col rounded-[calc(1rem-2px)] bg-gray-50 p-4">
          {/* Title */}
          <Text variant="h2" data-testid="agent-title" className="mb-3 w-full">
            {name}
          </Text>

          {/* Creator */}
          <div
            className="mb-3 flex w-full items-center gap-2 lg:mb-12"
            data-testid="agent-creator"
          >
            <Avatar className="h-7 w-7 shrink-0">
              {creatorAvatar && (
                <AvatarImage src={creatorAvatar} alt={`${creator} avatar`} />
              )}
              <AvatarFallback size={28}>{creator.charAt(0)}</AvatarFallback>
            </Avatar>
            <Text variant="body" className="text-md">
              by
            </Text>
            <Link
              href={`/marketplace/creator/${encodeURIComponent(creatorSlug ?? creator)}`}
              className="text-md font-medium hover:underline"
            >
              {creator}
            </Link>
          </div>

          {/* Short Description */}
          <div className="mb-4 line-clamp-2 w-full text-base font-normal leading-normal text-neutral-600 dark:text-neutral-300 sm:text-lg lg:mb-5 lg:text-xl lg:leading-7">
            {shortDescription}
          </div>

          {/* Buttons + Runs */}
          <div className="mt-6 flex w-full items-center justify-between lg:mt-8">
            <div className="flex gap-3">
              {user && (
                <Button
                  variant="primary"
                  className="group/add min-w-36 border-violet-600 bg-violet-600 transition-shadow duration-300 hover:border-violet-500 hover:bg-violet-500 hover:shadow-[0_0_20px_rgba(139,92,246,0.4)]"
                  data-testid="agent-add-library-button"
                  disabled={isAddingAgentToLibrary}
                  loading={isAddingAgentToLibrary}
                  leftIcon={
                    !isAddingAgentToLibrary && !isAgentAddedToLibrary ? (
                      <PlusIcon
                        size={16}
                        weight="bold"
                        className="transition-transform duration-300 group-hover/add:rotate-90 group-hover/add:scale-125"
                      />
                    ) : undefined
                  }
                  onClick={() =>
                    handleLibraryAction({
                      isAddingAgentFirstTime: !isAgentAddedToLibrary,
                    })
                  }
                >
                  {isAddingAgentToLibrary
                    ? "Adding..."
                    : isAgentAddedToLibrary
                      ? "See runs"
                      : "Add to library"}
                </Button>
              )}
              <Button
                variant="ghost"
                loading={isDownloadingAgent}
                onClick={() => handleDownload(agentId, name)}
                data-testid="agent-download-button"
              >
                {!isDownloadingAgent && <FileArrowDownIcon size={18} />}
                {isDownloadingAgent ? "Downloading..." : "Download"}
              </Button>
            </div>
            <Text
              variant="small"
              className="mr-4 hidden whitespace-nowrap text-zinc-500 lg:block"
            >
              {runs === 0
                ? "No runs"
                : `${runs.toLocaleString()} run${runs > 1 ? "s" : ""}`}
            </Text>
          </div>
        </div>
      </div>
      <div className="mb-8 flex flex-col gap-24">
        {/* Agent Details Section */}
        <div className="flex w-full flex-col gap-4 lg:gap-6">
          {/* Description Section */}
          <div className="mb-4 w-full">
            <Text variant="h5" className="mb-1.5 text-[1.3rem]">
              Description
            </Text>
            <Text
              variant="body"
              data-testid={"agent-description"}
              className="text-md whitespace-pre-line text-neutral-600"
            >
              {longDescription}
            </Text>
          </div>

          {/* Categories */}
          <div className="mb-4 flex w-full flex-col gap-1.5 sm:gap-2 md:px-2">
            <Text variant="h5" className="mb-1.5 text-[1.3rem]">
              Categories
            </Text>
            {categories.filter((c) => c.trim()).length > 0 ? (
              <div className="flex flex-wrap gap-1.5 sm:gap-2">
                {categories
                  .filter((c) => c.trim())
                  .map((category, index) => (
                    <Badge
                      variant="info"
                      key={index}
                      className="border border-purple-100 bg-purple-50 text-purple-800"
                    >
                      {category}
                    </Badge>
                  ))}
              </div>
            ) : (
              <Text variant="body" className="text-neutral-400">
                None
              </Text>
            )}
          </div>

          {/* Version history */}
          <div className="flex w-full flex-col gap-1.5 sm:gap-2 md:px-2">
            <div className="flex items-baseline justify-start">
              <Text variant="h5" className="text-[1.3rem]">
                Version
              </Text>
              {allVersions.length > 0 && (
                <Dialog
                  title="Changelog"
                  styling={{
                    maxWidth: "30rem",
                  }}
                >
                  <Dialog.Trigger>
                    <Button
                      variant="ghost"
                      size="small"
                      className="text-violet-600 hover:text-violet-500"
                    >
                      (Changelog)
                    </Button>
                  </Dialog.Trigger>
                  <Dialog.Content>
                    <div className="max-h-[60vh] space-y-4 overflow-y-auto p-4">
                      {allVersions.map(renderVersionItem)}
                    </div>
                  </Dialog.Content>
                </Dialog>
              )}
            </div>
            <div className="flex w-full items-center justify-start gap-8">
              <Text variant="body" className="text-neutral-600">
                {version}.0
              </Text>
              <Text variant="body" className="text-neutral-600">
                Last updated {formatTimeAgo(lastUpdated)}
              </Text>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
