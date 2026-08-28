"use client";

import {
  deleteV2DeleteLibraryAgent,
  getGetV2ListLibraryAgentsQueryKey,
  postV2AddMarketplaceAgent,
  useGetV2ListLibraryAgents,
} from "@/app/api/__generated__/endpoints/library/library";
import { getV2GetSpecificAgent } from "@/app/api/__generated__/endpoints/store/store";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { Button } from "@/components/atoms/Button/Button";
import type { ButtonProps } from "@/components/atoms/Button/helpers";
import {
  SplitButton,
  SplitButtonItem,
} from "@/components/atoms/SplitButton/SplitButton";
import {
  CreateSurface,
  getLastUsedTeam,
  getTeamScopedQueryKey,
  getTenantRequestInit,
  setLastUsedTeam,
} from "@/components/contextual/TeamPicker/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { analytics } from "@/services/analytics";
import { useOrgTeamStore } from "@/services/org-team/store";
import { getLibraryAgentHref } from "@/services/org-team/builder";
import * as Sentry from "@sentry/nextjs";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
  variant?: ButtonProps["variant"];
  size?: ButtonProps["size"];
}

const ORG_TARGET_LABEL = "Organization";

export function AddToLibraryButton({
  creatorSlug,
  agentSlug,
  agentName,
  agentGraphID,
  className,
  variant = "ghost",
  size = "small",
}: Props) {
  const { isLoggedIn } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const teams = useOrgTeamStore((s) => s.teams);
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const isLoaded = useOrgTeamStore((s) => s.isLoaded);
  const currentTeams = teams.filter((team) => team.orgId === activeOrgID);
  const libraryListParams = { is_hidden: false, page_size: 1000 };

  const { data: libraryAgents, isLoading: isLibraryLoading } =
    useGetV2ListLibraryAgents(libraryListParams, {
      request: getTenantRequestInit(activeOrgID, null, isLoaded),
      query: {
        queryKey: getTeamScopedQueryKey(
          getGetV2ListLibraryAgentsQueryKey(libraryListParams),
          activeOrgID,
          null,
        ),
        enabled: isLoggedIn && isLoaded,
        select: (res) =>
          res.status === 200 ? (res.data as LibraryAgentResponse) : undefined,
      },
    });

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
        getTenantRequestInit(activeOrgID, teamId),
      );
      return res.data as LibraryAgent;
    },
  });

  const { mutateAsync: removeFromLibrary } = useMutation({
    mutationFn: async (agent: LibraryAgent) =>
      deleteV2DeleteLibraryAgent(
        agent.id,
        getTenantRequestInit(
          agent.organization_id ?? null,
          agent.team_id ?? null,
        ),
      ),
  });

  if (!isLoggedIn) return null;

  const targets: { id: string | null; name: string }[] = [
    { id: null, name: ORG_TARGET_LABEL },
    ...currentTeams.map((team) => ({ id: team.id, name: team.name })),
  ];
  const availableTargets = targets.filter(
    (target) =>
      !libraryAgents?.agents?.some(
        (agent) =>
          agent.graph_id === agentGraphID &&
          (agent.organization_id ?? null) === (activeOrgID ?? null) &&
          (agent.team_id ?? null) === target.id,
      ),
  );

  if (!isLibraryLoading && availableTargets.length === 0) return null;

  async function handleAdd(teamId: string | null, e?: React.MouseEvent) {
    e?.stopPropagation();
    e?.preventDefault();

    try {
      const data = await addToLibrary({ teamId });
      // Only remember this target once the add actually succeeds, so a failed
      // request can't leave the split button defaulting to a team/Organization
      // that never received the agent.
      if (activeOrgID) {
        setLastUsedTeam(activeOrgID, CreateSurface.MarketplaceAdd, teamId);
      }

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
            libraryHref={getLibraryAgentHref(
              data.id,
              data.organization_id ?? activeOrgID ?? null,
              data.team_id ?? teamId,
            )}
            onUndo={async () => {
              try {
                await removeFromLibrary(data);
                await queryClient.invalidateQueries({
                  queryKey: getGetV2ListLibraryAgentsQueryKey(),
                });
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

  const buttonClassName =
    variant === "ghost"
      ? `z-10 text-zinc-500 hover:border-transparent hover:bg-transparent hover:text-zinc-800 ${className ?? ""}`
      : className;

  // Solo users (no teams): original plain button, adds to their org context.
  // While the org/team store is still loading we render the same control but
  // disable it, so a team member can't click the solo button during the async
  // load window and accidentally add to org context instead of a team.
  if (!isLoaded || currentTeams.length === 0) {
    return (
      <Button
        variant={variant}
        size={size}
        loading={isPending}
        disabled={!isLoaded || isLibraryLoading}
        leftIcon={<Icon icon={PlusSignIcon} size={14} />}
        onClick={(e) => handleAdd(null, e)}
        className={buttonClassName}
        aria-label={`Add ${agentName} to library`}
      >
        {isPending ? "Adding..." : "Add"}
      </Button>
    );
  }

  // Primary target = last-used, clamped to a still-valid team (else Organization).
  const lastUsedId = activeOrgID
    ? getLastUsedTeam(activeOrgID, CreateSurface.MarketplaceAdd)
    : null;
  const primaryTarget =
    availableTargets.find((target) => target.id === lastUsedId) ??
    availableTargets[0];
  const primaryTeamId = primaryTarget?.id ?? null;
  const primaryLabelName = primaryTarget?.name ?? ORG_TARGET_LABEL;

  const menuItems: SplitButtonItem[] = availableTargets
    .filter((target) => target.id !== primaryTeamId)
    .map((target) => ({
      key: target.id ?? "org-home",
      label: `Add to ${target.name}`,
      onSelect: () => handleAdd(target.id),
    }));

  return (
    <span onClick={(e) => e.stopPropagation()} className="z-10 inline-flex">
      <SplitButton
        variant={variant}
        size={size}
        loading={isPending}
        disabled={isLibraryLoading}
        leftIcon={<Icon icon={PlusSignIcon} size={14} />}
        primaryLabel={isPending ? "Adding..." : `Add to ${primaryLabelName}`}
        primaryAriaLabel={`Add ${agentName} to ${primaryLabelName}`}
        menuAriaLabel={`Choose where to add ${agentName}`}
        onPrimaryClick={(e) => handleAdd(primaryTeamId, e)}
        items={menuItems}
        buttonClassName={buttonClassName}
      />
    </span>
  );
}
