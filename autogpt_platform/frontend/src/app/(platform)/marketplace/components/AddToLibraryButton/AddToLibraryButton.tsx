"use client";

import {
  getGetV2ListLibraryAgentsQueryKey,
  postV2AddMarketplaceAgent,
  useDeleteV2DeleteLibraryAgent,
  useGetV2ListLibraryAgents,
} from "@/app/api/__generated__/endpoints/library/library";
import { getV2GetSpecificAgent } from "@/app/api/__generated__/endpoints/store/store";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { Button } from "@/components/atoms/Button/Button";
import {
  SplitButton,
  SplitButtonItem,
} from "@/components/atoms/SplitButton/SplitButton";
import {
  CreateSurface,
  getLastUsedTeam,
  getTeamRequestInit,
  setLastUsedTeam,
} from "@/components/contextual/TeamPicker/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { analytics } from "@/services/analytics";
import { useOrgTeamStore } from "@/services/org-team/store";
import { PlusIcon } from "@phosphor-icons/react";
import * as Sentry from "@sentry/nextjs";
import { useMutation, useQueryClient } from "@tanstack/react-query";
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

const ORG_TARGET_LABEL = "Organization";

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
  const teams = useOrgTeamStore((s) => s.teams);
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

  // Per-team header is chosen at click time, so add via the raw endpoint fn
  // (the generated hook binds its request options once, at hook creation).
  const { mutateAsync: addToLibrary, isPending } = useMutation({
    mutationFn: async ({ teamId }: { teamId: string | null }) => {
      const details = await getV2GetSpecificAgent(creatorSlug, agentSlug);
      if (details.status !== 200) {
        throw new Error("Failed to fetch agent details");
      }
      const res = await postV2AddMarketplaceAgent(
        { store_listing_version_id: details.data.store_listing_version_id },
        getTeamRequestInit(teamId),
      );
      return res.data as LibraryAgent;
    },
  });

  const { mutateAsync: removeFromLibrary } = useDeleteV2DeleteLibraryAgent();

  if (!isLoggedIn) return null;
  if (justAdded) return null;

  const isAlreadyInLibrary =
    isInLibrary ??
    libraryAgents?.agents?.some(
      (a: LibraryAgent) => a.graph_id === agentGraphID,
    );

  if (isAlreadyInLibrary) return null;

  async function handleAdd(teamId: string | null, e?: React.MouseEvent) {
    e?.stopPropagation();
    e?.preventDefault();

    try {
      setLastUsedTeam(CreateSurface.MarketplaceAdd, teamId);
      const data = await addToLibrary({ teamId });
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

  const ghostButtonClassName = `z-10 text-zinc-500 hover:border-transparent hover:bg-transparent hover:text-zinc-800 ${className ?? ""}`;

  // Solo users (no teams): original plain button, adds to their org context.
  if (teams.length === 0) {
    return (
      <Button
        variant="ghost"
        size="small"
        loading={isPending}
        leftIcon={<PlusIcon size={14} weight="bold" />}
        onClick={(e) => handleAdd(null, e)}
        className={ghostButtonClassName}
        aria-label={`Add ${agentName} to library`}
      >
        {isPending ? "Adding..." : "Add"}
      </Button>
    );
  }

  // Primary target = last-used, clamped to a still-valid team (else Organization).
  const lastUsedId = getLastUsedTeam(CreateSurface.MarketplaceAdd);
  const primaryTeam = teams.find((t) => t.id === lastUsedId) ?? null;
  const primaryTeamId = primaryTeam?.id ?? null;
  const primaryLabelName = primaryTeam?.name ?? ORG_TARGET_LABEL;

  const targets: { id: string | null; name: string }[] = [
    { id: null, name: ORG_TARGET_LABEL },
    ...teams.map((t) => ({ id: t.id, name: t.name })),
  ];

  const menuItems: SplitButtonItem[] = targets
    .filter((target) => target.id !== primaryTeamId)
    .map((target) => ({
      key: target.id ?? "org-home",
      label: `Add to ${target.name}`,
      onSelect: () => handleAdd(target.id),
    }));

  return (
    <span onClick={(e) => e.stopPropagation()} className="z-10 inline-flex">
      <SplitButton
        variant="ghost"
        size="small"
        loading={isPending}
        leftIcon={<PlusIcon size={14} weight="bold" />}
        primaryLabel={isPending ? "Adding..." : `Add to ${primaryLabelName}`}
        primaryAriaLabel={`Add ${agentName} to ${primaryLabelName}`}
        menuAriaLabel={`Choose where to add ${agentName}`}
        onPrimaryClick={(e) => handleAdd(primaryTeamId, e)}
        items={menuItems}
        buttonClassName={ghostButtonClassName}
      />
    </span>
  );
}
