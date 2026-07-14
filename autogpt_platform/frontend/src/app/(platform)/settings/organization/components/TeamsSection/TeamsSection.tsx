"use client";

import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { DotsThree, PlusIcon } from "@phosphor-icons/react";

import { CreateTeamDialog } from "./components/CreateTeamDialog/CreateTeamDialog";
import { TeamManagePanel } from "./components/TeamManagePanel/TeamManagePanel";
import { useTeamsSection } from "./useTeamsSection";

interface Props {
  orgId: string;
  orgMembers: OrgMemberResponse[];
  currentMember: OrgMemberResponse | null;
}

export function TeamsSection({ orgId, orgMembers, currentMember }: Props) {
  const {
    teams,
    isLoading,
    expandedTeamId,
    setExpandedTeamId,
    teamToDelete,
    setTeamToDelete,
    isDeleting,
    handleJoin,
    handleDeleteConfirmed,
    isCreateOpen,
    setIsCreateOpen,
    refetch,
  } = useTeamsSection({ orgId });

  const canCreate = Boolean(
    currentMember?.is_owner ||
      currentMember?.is_admin ||
      currentMember?.is_billing_manager,
  );
  const canDelete = Boolean(currentMember?.is_owner || currentMember?.is_admin);

  return (
    <section className="flex flex-col gap-4" data-testid="org-teams-section">
      <div className="flex items-center justify-between">
        <Text variant="h4" as="h2">
          Teams ({teams.length})
        </Text>
        {canCreate ? (
          <Button
            variant="secondary"
            size="small"
            onClick={() => setIsCreateOpen(true)}
            data-testid="create-team-button"
          >
            <PlusIcon size={14} />
            New team
          </Button>
        ) : null}
      </div>

      {isLoading ? (
        <Text variant="small" className="text-zinc-500">
          Loading teams…
        </Text>
      ) : teams.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          This organization has no teams yet.
        </Text>
      ) : (
        <ul className="flex flex-col divide-y divide-zinc-100">
          {teams.map((team) => {
            const isExpanded = expandedTeamId === team.id;
            return (
              <li
                key={team.id}
                className="flex flex-col gap-3 py-3"
                data-testid="org-team-row"
              >
                <div className="flex items-center gap-3">
                  <div className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate text-sm font-medium">
                      {team.name}
                    </span>
                    <span className="text-xs text-zinc-500">
                      {team.member_count}{" "}
                      {team.member_count === 1 ? "member" : "members"}
                    </span>
                  </div>
                  {team.is_default ? (
                    <Badge variant="info">Default</Badge>
                  ) : team.join_policy !== "OPEN" ? (
                    <Badge variant="info">Private</Badge>
                  ) : null}
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        className="rounded p-1.5 text-neutral-600 transition-colors hover:bg-neutral-100"
                        aria-label={`Team actions for ${team.name}`}
                        data-testid="team-actions-button"
                      >
                        <DotsThree className="h-5 w-5" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      {!team.is_default && team.join_policy === "OPEN" ? (
                        <DropdownMenuItem
                          onClick={() => handleJoin(team)}
                          data-testid="team-join-item"
                        >
                          Join
                        </DropdownMenuItem>
                      ) : null}
                      <DropdownMenuItem
                        onClick={() => setExpandedTeamId(team.id)}
                        data-testid="manage-team-button"
                      >
                        Manage
                      </DropdownMenuItem>
                      {canDelete && !team.is_default ? (
                        <DropdownMenuItem
                          onClick={() => setTeamToDelete(team)}
                          className="text-red-600 focus:bg-red-50 focus:text-red-600"
                          data-testid="team-delete-item"
                        >
                          Delete
                        </DropdownMenuItem>
                      ) : null}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                {isExpanded ? (
                  <TeamManagePanel
                    orgId={orgId}
                    team={team}
                    orgMembers={orgMembers}
                    currentUserId={currentMember?.user_id}
                    onChanged={refetch}
                    onDone={() => setExpandedTeamId(null)}
                    onLeft={() => {
                      setExpandedTeamId(null);
                      refetch();
                    }}
                  />
                ) : null}
              </li>
            );
          })}
        </ul>
      )}

      <CreateTeamDialog
        orgId={orgId}
        open={isCreateOpen}
        onOpenChange={setIsCreateOpen}
        onCreated={refetch}
      />

      <Dialog
        title="Delete team"
        styling={{ maxWidth: "26rem" }}
        controlled={{
          isOpen: Boolean(teamToDelete),
          set: (open) => {
            if (!open && !isDeleting) setTeamToDelete(null);
          },
        }}
      >
        <Dialog.Content>
          <Text variant="body">
            Delete {teamToDelete?.name}? This removes the team and its members
            from the organization. This can’t be undone.
          </Text>
          <div className="flex justify-end gap-2 pt-4">
            <Button
              variant="secondary"
              onClick={() => setTeamToDelete(null)}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              loading={isDeleting}
              onClick={handleDeleteConfirmed}
            >
              Delete team
            </Button>
          </div>
        </Dialog.Content>
      </Dialog>
    </section>
  );
}
