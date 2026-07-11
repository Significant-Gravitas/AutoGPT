"use client";

import { useState } from "react";

import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { TeamProfileForm } from "./TeamProfileForm";
import { useTeamManagePanel } from "./useTeamManagePanel";

const ROLE_OPTIONS = [
  { value: "member", label: "Member" },
  { value: "admin", label: "Admin" },
];

interface Props {
  orgId: string;
  team: TeamResponse;
  orgMembers: OrgMemberResponse[];
  currentUserId: string | undefined;
  onChanged: () => void;
  onLeft: () => void;
}

export function TeamManagePanel({
  orgId,
  team,
  orgMembers,
  currentUserId,
  onChanged,
  onLeft,
}: Props) {
  const [selectedUserId, setSelectedUserId] = useState("");
  const {
    members,
    isLoading,
    isTeamAdmin,
    memberToRemove,
    setMemberToRemove,
    isLeaveOpen,
    setIsLeaveOpen,
    isAdding,
    isUpdatingRole,
    isRemoving,
    isLeaving,
    handleAddMember,
    handleRoleChange,
    handleRemoveConfirmed,
    handleLeaveConfirmed,
  } = useTeamManagePanel({
    orgId,
    wsId: team.id,
    currentUserId,
    onChanged,
    onLeft,
  });

  const availableMembers = orgMembers.filter(
    (om) => !members.some((tm) => tm.user_id === om.user_id),
  );
  const addOptions = availableMembers.map((om) => ({
    value: om.user_id,
    label: om.name || om.email,
  }));

  function handleAdd() {
    const picked = addOptions.find((o) => o.value === selectedUserId);
    if (!picked) return;
    handleAddMember(picked.value, picked.label);
    setSelectedUserId("");
  }

  return (
    <div
      className="flex flex-col gap-4 rounded-lg bg-zinc-50 p-4"
      data-testid="team-manage-panel"
    >
      {isTeamAdmin ? (
        <TeamProfileForm orgId={orgId} team={team} onSaved={onChanged} />
      ) : null}

      <div className="flex flex-col gap-2">
        <Text variant="large-medium" as="h4">
          Team members ({members.length})
        </Text>
        {isLoading ? (
          <Text variant="small" className="text-zinc-500">
            Loading members…
          </Text>
        ) : (
          <ul className="flex flex-col divide-y divide-zinc-100">
            {members.map((member) => {
              const isSelf = member.user_id === currentUserId;
              return (
                <li
                  key={member.user_id}
                  className="flex items-center gap-3 py-2"
                  data-testid="team-member-row"
                >
                  <div className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate text-sm font-medium">
                      {member.name || member.email}
                      {isSelf ? " (you)" : ""}
                    </span>
                    <span className="truncate text-xs text-zinc-500">
                      {member.email}
                    </span>
                  </div>
                  {isTeamAdmin && !isSelf ? (
                    <>
                      <Select
                        id={`team-role-${team.id}-${member.user_id}`}
                        label=""
                        hideLabel
                        size="small"
                        wrapperClassName="!mb-0 w-32"
                        value={member.is_admin ? "admin" : "member"}
                        onValueChange={(role) => handleRoleChange(member, role)}
                        options={ROLE_OPTIONS}
                        disabled={isUpdatingRole}
                      />
                      <Button
                        variant="ghost"
                        size="small"
                        onClick={() => setMemberToRemove(member)}
                      >
                        Remove
                      </Button>
                    </>
                  ) : member.is_admin ? (
                    <Badge variant="info">Admin</Badge>
                  ) : null}
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {isTeamAdmin && addOptions.length > 0 ? (
        <div className="flex items-end gap-2">
          <Select
            id={`team-add-member-${team.id}`}
            label="Add a member"
            placeholder="Select an org member"
            wrapperClassName="!mb-0 w-64"
            value={selectedUserId}
            onValueChange={setSelectedUserId}
            options={addOptions}
          />
          <Button
            variant="secondary"
            onClick={handleAdd}
            loading={isAdding}
            disabled={!selectedUserId}
          >
            Add
          </Button>
        </div>
      ) : null}

      {!isTeamAdmin && !team.is_default ? (
        <div>
          <Button
            variant="secondary"
            onClick={() => setIsLeaveOpen(true)}
            data-testid="team-leave-button"
          >
            Leave team
          </Button>
        </div>
      ) : null}

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
