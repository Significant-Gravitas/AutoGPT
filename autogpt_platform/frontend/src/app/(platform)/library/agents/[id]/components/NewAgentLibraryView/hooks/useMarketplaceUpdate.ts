import { useGetV2GetSpecificAgent } from "@/app/api/__generated__/endpoints/store/store";
import {
  usePatchV2UpdateLibraryAgent,
  getGetV2GetLibraryAgentQueryKey,
} from "@/app/api/__generated__/endpoints/library/library";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useQueryClient } from "@tanstack/react-query";
import * as React from "react";
import { useState } from "react";

interface UseMarketplaceUpdateProps {
  agent: LibraryAgent | null | undefined;
}

export function useMarketplaceUpdate({ agent }: UseMarketplaceUpdateProps) {
  const [modalOpen, setModalOpen] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Get marketplace data if agent has marketplace listing
  const { data: storeAgentData } = useGetV2GetSpecificAgent(
    agent?.marketplace_listing?.creator.slug || "",
    agent?.marketplace_listing?.slug || "",
    {
      query: {
        enabled: !!(
          agent?.marketplace_listing?.creator.slug &&
          agent?.marketplace_listing?.slug
        ),
      },
    },
  );

  const updateToLatestMutation = usePatchV2UpdateLibraryAgent({
    mutation: {
      onError: (err) => {
        toast({
          title: "Update Failed",
          description: "Failed to update agent to latest version",
          variant: "destructive",
        });
        console.error("Failed to update agent:", err);
      },
      onSuccess: () => {
        toast({
          title: "Agent Updated",
          description: "Agent updated to latest version successfully",
        });
        // Invalidate to get the updated agent data from the server
        if (agent?.id) {
          queryClient.invalidateQueries({
            queryKey: getGetV2GetLibraryAgentQueryKey(agent.id),
          });
        }
      },
    },
  });

  // Check if marketplace has a newer version than user's current version
  const marketplaceUpdateInfo = React.useMemo(() => {
    if (!agent || !storeAgentData || storeAgentData.status !== 200) {
      return {
        hasUpdate: false,
        latestVersion: undefined,
        isUserCreator: false,
      };
    }

    const storeAgent = storeAgentData.data;

    // Get the latest version from the marketplace
    // agentGraphVersions array contains graph version numbers as strings, get the highest one
    const latestMarketplaceVersion =
      storeAgent.agentGraphVersions.length > 0
        ? Math.max(...storeAgent.agentGraphVersions.map((v) => parseInt(v, 10)))
        : undefined;

    // For now, we treat all users as non-creators for marketplace updates
    // TODO: Implement proper creator detection if needed
    const isUserCreator = false;

    // If marketplace version is newer than user's version, show update banner
    const hasMarketplaceUpdate =
      latestMarketplaceVersion !== undefined &&
      latestMarketplaceVersion > agent.graph_version;

    return {
      hasUpdate: hasMarketplaceUpdate,
      latestVersion: latestMarketplaceVersion,
      isUserCreator,
      hasPublishUpdate: false, // TODO: Implement creator publish update detection
    };
  }, [agent, storeAgentData]);

  const handlePublishUpdate = () => {
    setModalOpen(true);
  };

  const handleUpdateToLatest = () => {
    if (!agent || marketplaceUpdateInfo.latestVersion === undefined) return;
    // Update to the specific marketplace version using the new graph_version parameter
    updateToLatestMutation.mutate({
      libraryAgentId: agent.id,
      data: {
        graph_version: marketplaceUpdateInfo.latestVersion,
      },
    });
  };

  return {
    hasAgentMarketplaceUpdate: marketplaceUpdateInfo.hasPublishUpdate,
    hasMarketplaceUpdate: marketplaceUpdateInfo.hasUpdate,
    latestMarketplaceVersion: marketplaceUpdateInfo.latestVersion,
    isUpdating: updateToLatestMutation.isPending,
    modalOpen,
    setModalOpen,
    handlePublishUpdate,
    handleUpdateToLatest,
  };
}
