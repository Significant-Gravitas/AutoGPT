"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { useGetV2GetSpecificAgent } from "@/app/api/__generated__/endpoints/store/store";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import type { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import React from "react";

interface AgentVersionChangelogProps {
  agent: LibraryAgent;
  isOpen: boolean;
  onClose: () => void;
}

interface VersionInfo {
  version: number;
  isCurrentVersion: boolean;
}

export function AgentVersionChangelog({
  agent,
  isOpen,
  onClose,
}: AgentVersionChangelogProps) {
  // Get marketplace data if agent has marketplace listing
  const { data: storeAgentData, isLoading } = useGetV2GetSpecificAgent(
    agent?.marketplace_listing?.creator.slug || "",
    agent?.marketplace_listing?.slug || "",
    {},
    {
      query: {
        enabled: !!(
          agent?.marketplace_listing?.creator.slug &&
          agent?.marketplace_listing?.slug
        ),
      },
    },
  );

  // Create version info from available graph versions
  const storeData = okData(storeAgentData) as StoreAgentDetails | undefined;
  const agentVersions: VersionInfo[] = storeData?.graph_versions
    ? storeData.graph_versions
        .map((versionStr: string) => parseInt(versionStr, 10))
        .sort((a: number, b: number) => b - a) // Sort descending (newest first)
        .map((version: number) => ({
          version,
          isCurrentVersion: version === agent.graph_version,
        }))
    : [];

  const renderVersionItem = (versionInfo: VersionInfo) => {
    return (
      <div
        key={versionInfo.version}
        className={`rounded-lg border p-4 ${
          versionInfo.isCurrentVersion
            ? // TODO(tokens): no exact role for border-blue-200 (#BFDBFE); no dark roles yet for dark:border-blue-800 (#1E40AF), dark:bg-blue-950 (#172554)
              "border-blue-200 bg-info-surface dark:border-blue-800 dark:bg-blue-950"
            : // TODO(tokens): no exact role for border-neutral-200 (#E5E5E5); no dark roles yet for dark:border-neutral-700 (#404040), dark:bg-neutral-800 (#262626)
              "border-neutral-200 bg-surface dark:border-neutral-700 dark:bg-neutral-800"
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Text variant="body-semibold">v{versionInfo.version}</Text>
            {versionInfo.isCurrentVersion && (
              <Text
                variant="small-medium-tight"
                as="span"
                // TODO(tokens): no exact role for bg-blue-100 (#DBEAFE), text-blue-800 (#1E40AF); no dark roles yet for dark:bg-blue-900 (#1E3A8A), dark:text-blue-100 (#DBEAFE)
                className="rounded-full bg-blue-100 px-2 py-1 text-blue-800 dark:bg-blue-900 dark:text-blue-100"
              >
                Current
              </Text>
            )}
          </div>
        </div>

        <Text
          variant="small"
          // TODO(tokens): no exact role for text-neutral-600 (#525252); no dark roles yet for dark:text-neutral-400 (#A3A3A3)
          className="mt-1 text-neutral-600 dark:text-neutral-400"
        >
          Available marketplace version
        </Text>
      </div>
    );
  };

  return (
    <Dialog
      title={`Version History - ${agent.name}`}
      styling={{
        maxWidth: "45rem",
      }}
      controlled={{
        isOpen: isOpen,
        set: (isOpen) => {
          if (!isOpen) {
            onClose();
          }
        },
      }}
    >
      <Dialog.Content>
        <div className="max-h-[70vh] overflow-y-auto">
          {isLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-20 w-full" />
              <Skeleton className="h-20 w-full" />
              <Skeleton className="h-20 w-full" />
            </div>
          ) : agentVersions.length > 0 ? (
            <div className="space-y-4">
              <Text
                variant="small"
                // TODO(tokens): no exact role for text-neutral-600 (#525252); no dark roles yet for dark:text-neutral-400 (#A3A3A3)
                className="text-neutral-600 dark:text-neutral-400"
              >
                View changes and updates across different versions of this
                workflow.
              </Text>
              {agentVersions.map(renderVersionItem)}
            </div>
          ) : (
            <div className="py-8 text-center">
              <Text
                variant="body"
                // TODO(tokens): no exact role for text-neutral-600 (#525252); no dark roles yet for dark:text-neutral-400 (#A3A3A3)
                className="text-neutral-600 dark:text-neutral-400"
              >
                No version history available for this workflow.
              </Text>
            </div>
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
