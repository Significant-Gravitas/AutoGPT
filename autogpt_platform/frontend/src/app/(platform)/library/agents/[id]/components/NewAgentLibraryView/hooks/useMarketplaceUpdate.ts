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

    const submissionsResponse = okData(submissionsData) as any;
    const agentSubmissions =
      submissionsResponse?.submissions?.filter(
        (submission: StoreSubmission) => submission.agent_id === agent.graph_id,
      ) || [];

    const highestSubmittedVersion =
      agentSubmissions.length > 0
        ? Math.max(
            ...agentSubmissions.map(
              (submission: StoreSubmission) => submission.agent_version,
            ),
          )
        : 0;

    const hasUnpublishedChanges =
      isUserCreator && agent.graph_version > highestSubmittedVersion;

    if (!storeAgent) {
      return {
        hasUpdate: false,
        latestVersion: undefined,
        isUserCreator,
        hasPublishUpdate: agentSubmissions.length > 0 && hasUnpublishedChanges,
      };
    }

    const latestMarketplaceVersion =
      storeAgent.agentGraphVersions?.length > 0
        ? Math.max(
            ...storeAgent.agentGraphVersions.map((v: string) =>
              parseInt(v, 10),
            ),
          )
        : undefined;

    const hasPublishUpdate =
      isUserCreator &&
      agent.graph_version >
        Math.max(latestMarketplaceVersion || 0, highestSubmittedVersion);

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
