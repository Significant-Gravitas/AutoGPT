import {
  useGetV2GetSpecificAgent,
  useGetV2ListMySubmissions,
} from "@/app/api/__generated__/endpoints/store/store";
import {
  usePatchV2UpdateLibraryAgent,
  getGetV2GetLibraryAgentQueryKey,
} from "@/app/api/__generated__/endpoints/library/library";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useQueryClient } from "@tanstack/react-query";
import { useSupabaseStore } from "@/lib/supabase/hooks/useSupabaseStore";
import { okData } from "@/app/api/helpers";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import * as React from "react";
import { useState } from "react";

interface UseMarketplaceUpdateProps {
  agent: LibraryAgent | null | undefined;
}

export function useMarketplaceUpdate({ agent }: UseMarketplaceUpdateProps) {
  const [modalOpen, setModalOpen] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const user = useSupabaseStore((state) => state.user);

  // Get marketplace data if agent has marketplace listing
  const { data: storeAgentData } = useGetV2GetSpecificAgent(
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

  // Get user's submissions - only fetch if user is the creator
  const { data: submissionsData, isLoading: isSubmissionsLoading } =
    useGetV2ListMySubmissions(
      { page: 1, page_size: 50 },
      {
        query: {
          // Only fetch if user is the creator
          enabled: !!(user?.id && agent?.owner_user_id === user.id),
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
    const storeAgent = okData(storeAgentData) as any;

    if (!agent || isSubmissionsLoading) {
      return {
        hasUpdate: false,
        latestVersion: undefined,
        isUserCreator: false,
        hasPublishUpdate: false,
      };
    }

    const isUserCreator = agent?.owner_user_id === user?.id;

    // Check if there's a pending submission for this specific agent version
    const submissionsResponse = okData(submissionsData) as any;
    const hasPendingSubmissionForCurrentVersion =
      isUserCreator &&
      submissionsResponse?.submissions?.some(
        (submission: StoreSubmission) =>
          submission.agent_id === agent.graph_id &&
          submission.agent_version === agent.graph_version &&
          submission.status === "PENDING",
      );

    if (!storeAgent) {
      return {
        hasUpdate: false,
        latestVersion: undefined,
        isUserCreator,
        hasPublishUpdate:
          isUserCreator && !hasPendingSubmissionForCurrentVersion,
      };
    }

    // Get the latest version from the marketplace
    // agentGraphVersions array contains graph version numbers as strings, get the highest one
    const latestMarketplaceVersion =
      storeAgent.agentGraphVersions?.length > 0
        ? Math.max(
            ...storeAgent.agentGraphVersions.map((v: string) =>
              parseInt(v, 10),
            ),
          )
        : undefined;

    // Show publish update button if:
    // 1. User is the creator
    // 2. No pending submission for current version
    // 3. Either: agent not published yet OR local version is newer than marketplace
    const hasPublishUpdate =
      isUserCreator &&
      !hasPendingSubmissionForCurrentVersion &&
      (latestMarketplaceVersion === undefined || // Not published yet
        agent.graph_version > latestMarketplaceVersion); // Or local version is newer

    // If marketplace version is newer than user's version, show update banner
    // This applies to both creators and non-creators
    const hasMarketplaceUpdate =
      latestMarketplaceVersion !== undefined &&
      latestMarketplaceVersion > agent.graph_version;

    return {
      hasUpdate: hasMarketplaceUpdate,
      latestVersion: latestMarketplaceVersion,
      isUserCreator,
      hasPublishUpdate,
    };
  }, [agent, storeAgentData, user, submissionsData, isSubmissionsLoading]);

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
