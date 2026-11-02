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
import type { User } from "@/lib/auth/types";
import Link from "next/link";
import { InstallOnExpertButton } from "../InstallOnExpertButton/InstallOnExpertButton";
import { useAgentInfo } from "./useAgentInfo";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface AgentInfoProps {
  user: User | null;
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
            <Text variant="body-semibold" tone="primary">
              Version {versionInfo.version}.0
            </Text>
            {versionInfo.isCurrentVersion && (
              <span className="rounded bg-info/20 px-1.5 py-0.5 text-xs font-medium text-info-foreground">
                Current
              </span>
            )}
          </div>
          {changelogEntry && (
            <Text variant="small" tone="muted">
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
            <Text variant="body" tone="body">
              {changelogEntry.changes_summary}
            </Text>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="w-full px-4 sm:px-6 lg:px-0">
      <div
        // oxlint-disable-next-line design-system/no-raw-colors -- indigo has no role; info or brand would change the frame's hue (deferred)
        className="mb-8 rounded-2xl bg-gradient-to-r from-info/30 to-indigo-200 p-px"
      >
        <div className="flex flex-col rounded-large bg-surface-muted p-4">
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
              <AvatarFallback size={28}>{creator}</AvatarFallback>
            </Avatar>
            <Text variant="body">by</Text>
            <Link
              href={`/marketplace/creator/${encodeURIComponent(creatorSlug ?? creator)}`}
              className="font-medium hover:underline"
            >
              {creator}
            </Link>
          </div>

          {/* Short Description */}
          <div className="mb-4 line-clamp-2 w-full text-base leading-normal font-normal text-foreground-secondary sm:text-lg lg:mb-5 lg:text-xl lg:leading-7">
            {shortDescription}
          </div>

          {/* Buttons */}
          <div className="mt-6 flex w-full items-center gap-2 lg:mt-8">
            {user && (
              <Button
                variant="primary"
                /* oxlint-disable design-system/no-restyle -- the purple add button; no Button variant is brand */
                /* oxlint-disable design-system/no-arbitrary-values -- the hover glow has no shadow token */
                className="group/add min-w-36 border-brand bg-brand transition-shadow duration-300 hover:border-brand/90 hover:bg-brand/90 hover:shadow-[0_0_20px_rgba(139,92,246,0.4)]"
                /* oxlint-enable design-system/no-arbitrary-values */
                /* oxlint-enable design-system/no-restyle */
                data-testid="agent-add-library-button"
                disabled={isAddingAgentToLibrary}
                loading={isAddingAgentToLibrary}
                leftIcon={
                  !isAddingAgentToLibrary && !isAgentAddedToLibrary ? (
                    <Icon
                      icon={PlusSignIcon}
                      size={16}
                      // oxlint-disable-next-line design-system/no-restyle -- the plus spins on hover; Icon has no motion prop
                      className="transition-transform duration-300 group-hover/add:scale-125 group-hover/add:rotate-90"
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
            <InstallOnExpertButton
              storeListingVersionId={storeListingVersionId}
            />
          </div>

          {/* Download */}
          <div className="mt-3 flex w-full items-center justify-between gap-2">
            <div className="flex items-center gap-0">
              <Text variant="body" tone="muted">
                Want to use this workflow locally?
              </Text>
              <Button
                variant="ghost"
                size="small"
                loading={isDownloadingAgent}
                onClick={() => handleDownload(name)}
                data-testid="agent-download-button"
                // oxlint-disable-next-line design-system/no-restyle -- an underlined ghost button; the link variant changes its colour and padding
                className="underline"
              >
                {isDownloadingAgent ? "Downloading..." : "Download here"}
              </Button>
            </div>
            <Text variant="body" className="shrink-0 whitespace-nowrap">
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
            <Text variant="h4" as="h5" className="mb-1.5">
              Description
            </Text>
            <Text
              variant="body"
              data-testid={"agent-description"}
              tone="secondary"
              className="whitespace-pre-line"
            >
              {longDescription}
            </Text>
          </div>

          {/* Categories */}
          <div className="mb-4 flex w-full flex-col gap-1.5 sm:gap-2 md:px-2">
            <Text variant="h4" as="h5" className="mb-1.5">
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
                      // oxlint-disable-next-line design-system/no-restyle -- the category chips keep their purple fill; Badge has no brand variant
                      className="border border-brand/10 bg-accent-surface text-accent-strong"
                    >
                      {category}
                    </Badge>
                  ))}
              </div>
            ) : (
              <Text variant="body" tone="subtle">
                None
              </Text>
            )}
          </div>

          {/* Version history */}
          <div className="flex w-full flex-col gap-1.5 sm:gap-2 md:px-2">
            <div className="flex items-baseline justify-start">
              <Text variant="h4" as="h5">
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
                      // oxlint-disable-next-line design-system/no-restyle -- a purple ghost link; no Button variant is brand
                      className="text-brand hover:text-brand-hover"
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
              <Text variant="body" tone="secondary">
                {version}.0
              </Text>
              <Text variant="body" tone="secondary">
                Last updated {formatTimeAgo(lastUpdated)}
              </Text>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
