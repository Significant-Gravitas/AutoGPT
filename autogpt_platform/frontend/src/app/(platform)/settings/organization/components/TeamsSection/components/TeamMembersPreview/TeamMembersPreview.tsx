"use client";

import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";
import Avatar, { AvatarFallback } from "@/components/atoms/Avatar/Avatar";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { DotsThree } from "@phosphor-icons/react";

import { useTeamMembersPreview } from "./useTeamMembersPreview";

interface Props {
  orgId: string;
  team: TeamResponse;
  orgIsAdmin: boolean;
  onChanged: () => void;
}

export function TeamMembersPreview({
  orgId,
  team,
  orgIsAdmin,
  onChanged,
}: Props) {
  const {
    members,
    isLoading,
    isError,
    isPrivate,
    currentUserId,
    canManage,
    memberToRemove,
    setMemberToRemove,
    isLeaveOpen,
    setIsLeaveOpen,
    isRemoving,
    isLeaving,
    handleToggleAdmin,
    handleRemoveConfirmed,
    handleLeaveConfirmed,
  } = useTeamMembersPreview({ orgId, team, orgIsAdmin, onChanged });

  return (
    <div
      id={`team-members-${team.id}`}
      className="rounded-lg bg-zinc-50 px-3 py-2"
      data-testid="team-members-preview"
    >
      {isLoading ? (
        <ul className="flex flex-col divide-y divide-zinc-100">
          {[0, 1].map((i) => (
            <li
              key={i}
              className="flex items-center gap-3 py-2"
              data-testid="team-member-skeleton"
            >
              <Skeleton className="size-8 shrink-0 rounded-full" />
              <div className="flex flex-1 flex-col gap-1.5">
                <Skeleton className="h-3 w-32" />
                <Skeleton className="h-3 w-44" />
              </div>
            </li>
          ))}
        </ul>
      ) : isError ? (
        <Text
          variant="small"
          className="text-zinc-500"
          data-testid="team-members-hint"
        >
          {isPrivate
            ? "Private — join this team to see its members."
            : "Couldn't load members."}
        </Text>
      ) : members.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          This team has no members yet.
        </Text>
      ) : (
        <ul className="flex flex-col divide-y divide-zinc-100">
          {members.map((member) => {
            const isSelf = member.user_id === currentUserId;
            return (
              <li
                key={member.user_id}
                className="flex items-center gap-3 py-2"
                data-testid="team-member-preview-row"
              >
                <Avatar className="size-8 shrink-0">
                  <AvatarFallback className="text-xs">
                    {(member.name || member.email).charAt(0).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                <div className="flex min-w-0 flex-1 flex-col">
                  <span className="truncate text-sm font-medium">
                    {member.name || member.email}
                    {isSelf ? " (you)" : ""}
                  </span>
                  <span className="truncate text-xs text-zinc-500">
                    {member.email}
                  </span>
                </div>
                {member.is_admin ? <Badge variant="info">Admin</Badge> : null}
                {isSelf ? (
                  !team.is_default ? (
                    <Button
                      variant="ghost"
                      size="small"
                      onClick={() => setIsLeaveOpen(true)}
                      data-testid="team-preview-leave-button"
                    >
                      Leave
                    </Button>
                  ) : null
                ) : canManage ? (
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        className="rounded p-1.5 text-neutral-600 transition-colors hover:bg-neutral-100"
                        aria-label={`Member actions for ${member.name || member.email}`}
                        data-testid="team-member-actions-button"
                      >
                        <DotsThree className="h-5 w-5" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={() => handleToggleAdmin(member)}
                        data-testid="team-member-role-item"
                      >
                        {member.is_admin
                          ? "Remove admin"
                          : "Promote to team admin"}
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => setMemberToRemove(member)}
                        className="text-red-600 focus:bg-red-50 focus:text-red-600"
                        data-testid="team-member-remove-item"
                      >
                        Remove from team
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}

      <Dialog
        title="Remove member"
        styling={{ maxWidth: "26rem" }}
        controlled={{
          isOpen: Boolean(memberToRemove),
          set: (open) => {
            if (!open && !isRemoving) setMemberToRemove(null);
          },
        }}
      >
        <Dialog.Content>
          <Text variant="body">
            Remove {memberToRemove?.name || memberToRemove?.email} from{" "}
            {team.name}?
          </Text>
          <div className="flex justify-end gap-2 pt-4">
            <Button
              variant="secondary"
              onClick={() => setMemberToRemove(null)}
              disabled={isRemoving}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              loading={isRemoving}
              onClick={handleRemoveConfirmed}
            >
              Remove member
            </Button>
          </div>
        </Dialog.Content>
      </Dialog>

      <Dialog
        title="Leave team"
        styling={{ maxWidth: "26rem" }}
        controlled={{
          isOpen: isLeaveOpen,
          set: (open) => {
            if (!open && !isLeaving) setIsLeaveOpen(false);
          },
        }}
      >
        <Dialog.Content>
          <Text variant="body">
            Leave {team.name}? You will lose access to its resources.
          </Text>
          <div className="flex justify-end gap-2 pt-4">
            <Button
              variant="secondary"
              onClick={() => setIsLeaveOpen(false)}
              disabled={isLeaving}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              loading={isLeaving}
              onClick={handleLeaveConfirmed}
            >
              Leave team
            </Button>
          </div>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}
