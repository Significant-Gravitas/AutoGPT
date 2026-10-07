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
            ? "border-blue-200 bg-blue-50"
            : "border-zinc-200 bg-white"
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Text variant="body" className="font-semibold">
              v{versionInfo.version}
            </Text>
            {versionInfo.isCurrentVersion && (
              <span className="rounded-full bg-blue-100 px-2 py-1 text-xs font-medium text-blue-800">
                Current
              </span>
            )}
          </div>
        </div>

        <Text variant="small" tone="secondary" className="mt-1">
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
              <Text variant="small" tone="secondary">
                View changes and updates across different versions of this
                agent.
              </Text>
              {agentVersions.map(renderVersionItem)}
            </div>
          ) : (
            <div className="py-8 text-center">
              <Text variant="body" tone="secondary">
                No version history available for this agent.
              </Text>
            </div>
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
