"use client";

import {
  getGetV2ListLibraryAgentsQueryKey,
  useDeleteV2DeleteLibraryAgent,
  useGetV2ListLibraryAgents,
  usePostV2AddMarketplaceAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import { getV2GetSpecificAgent } from "@/app/api/__generated__/endpoints/store/store";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { Button } from "@/components/atoms/Button/Button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { analytics } from "@/services/analytics";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  creatorSlug: string;
  agentSlug: string;
  agentName: string;
  agentGraphID: string;
  className?: string;
  isInLibrary?: boolean;
}

export function AddToLibraryButton({
  creatorSlug,
  agentSlug,
  agentName,
  agentGraphID,
  className,
  isInLibrary,
}: Props) {
  const { isLoggedIn } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [justAdded, setJustAdded] = useState(false);

  // Only fetch library list if isInLibrary wasn't provided by parent
  const { data: libraryAgents } = useGetV2ListLibraryAgents(
    { is_hidden: false },
    {
      query: {
        enabled: isLoggedIn && isInLibrary === undefined,
        select: (res) =>
          res.status === 200 ? (res.data as LibraryAgentResponse) : undefined,
      },
    },
  );

  const { mutateAsync: addToLibrary, isPending } =
    usePostV2AddMarketplaceAgent();

  const { mutateAsync: removeFromLibrary } = useDeleteV2DeleteLibraryAgent();

  if (!isLoggedIn) return null;
  if (justAdded) return null;

  const isAlreadyInLibrary =
    isInLibrary ??
    libraryAgents?.agents?.some(
      (a: LibraryAgent) => a.graph_id === agentGraphID,
    );

  if (isAlreadyInLibrary) return null;

  async function handleClick(e: React.MouseEvent) {
    e.stopPropagation();
    e.preventDefault();

    try {
      const details = await getV2GetSpecificAgent(creatorSlug, agentSlug);

      if (details.status !== 200) {
        throw new Error("Failed to fetch agent details");
      }

      const { data: response } = await addToLibrary({
        data: {
          store_listing_version_id: details.data.store_listing_version_id,
        },
      });

      const data = response as LibraryAgent;
      setJustAdded(true);

      await queryClient.invalidateQueries({
        queryKey: getGetV2ListLibraryAgentsQueryKey(),
      });

      analytics.sendDatafastEvent("add_to_library", {
        name: data.name,
        id: data.id,
      });

      toast({
        title: `Agent ${agentName} added to your library.`,
        description: "Open it from your library, or undo.",
        action: (
          <Button
            onClick={async () => {
              try {
                await removeFromLibrary({ libraryAgentId: data.id });
                await queryClient.invalidateQueries({
                  queryKey: getGetV2ListLibraryAgentsQueryKey(),
                });
                setJustAdded(false);
                toast({
                  title: "Action undone.",
                  variant: "info",
                  duration: 3000,
                });
              } catch (undoError) {
                Sentry.captureException(undoError);
                toast({
                  title: "Failed to undo. Please try again.",
                  variant: "destructive",
                });
              }
            }}
          >
            Undo
          </Button>
        ),
        duration: 10000,
      });
    } catch (error) {
      Sentry.captureException(error);
      toast({
        title: "Error",
        description: "Failed to add agent to library. Please try again.",
        variant: "destructive",
      });
    }
  }

  return (
    <Button
      variant="secondary"
      size="md"
      loading={isPending}
      leftIcon={<Icon icon={PlusSignIcon} size={14} />}
      onClick={handleClick}
      className={`z-10 ${className ?? ""}`}
      aria-label={`Add ${agentName} to library`}
    >
      {isPending ? "Adding..." : "Add"}
    </Button>
  );
}
