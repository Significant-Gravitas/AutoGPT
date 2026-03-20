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
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { analytics } from "@/services/analytics";
import { PlusIcon } from "@phosphor-icons/react";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

function UndoActions({
  libraryAgentID,
  libraryHref,
  onUndo,
}: {
  libraryAgentID: string;
  libraryHref: string;
  onUndo: (id: string) => Promise<void>;
}) {
  const [isUndoing, setIsUndoing] = useState(false);

  return (
    <div className="mt-6 flex items-center gap-2">
      <Button
        variant="primary"
        size="small"
        as="NextLink"
        className="bg-neutral-200 text-zinc-900 hover:bg-neutral-300 hover:text-zinc-800"
        href={libraryHref}
      >
        Open agent
      </Button>
      <Button
        variant="ghost"
        size="small"
        loading={isUndoing}
        className="border-none text-zinc-200 hover:bg-transparent hover:text-zinc-400"
        onClick={async () => {
          setIsUndoing(true);
          try {
            await onUndo(libraryAgentID);
          } finally {
            setIsUndoing(false);
          }
        }}
      >
        {isUndoing ? "Undoing..." : "Undo"}
      </Button>
    </div>
  );
}

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
  const { isLoggedIn } = useSupabase();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [justAdded, setJustAdded] = useState(false);

  // Only fetch library list if isInLibrary wasn't provided by parent
  const { data: libraryAgents } = useGetV2ListLibraryAgents(undefined, {
    query: {
      enabled: isLoggedIn && isInLibrary === undefined,
      select: (res) =>
        res.status === 200 ? (res.data as LibraryAgentResponse) : undefined,
    },
  });

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

      const addedToast = toast({
        title: `Agent ${agentName} added to your library.`,
        description: (
          <UndoActions
            libraryAgentID={data.id}
            libraryHref={`/library/agents/${data.id}`}
            onUndo={async (id) => {
              try {
                await removeFromLibrary({ libraryAgentId: id });
                await queryClient.invalidateQueries({
                  queryKey: getGetV2ListLibraryAgentsQueryKey(),
                });
                setJustAdded(false);
                addedToast.dismiss();
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
          />
        ),
        dismissable: false,
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
      variant="ghost"
      size="small"
      loading={isPending}
      leftIcon={<PlusIcon size={14} weight="bold" />}
      onClick={handleClick}
      className={`z-10 text-zinc-500 hover:border-transparent hover:bg-transparent hover:text-zinc-800 ${className ?? ""}`}
      aria-label={`Add ${agentName} to library`}
    >
      {isPending ? "Adding..." : "Add"}
    </Button>
  );
}
