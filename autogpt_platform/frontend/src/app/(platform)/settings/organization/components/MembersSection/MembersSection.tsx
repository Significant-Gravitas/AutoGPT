"use client";

import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import Avatar, { AvatarFallback } from "@/components/atoms/Avatar/Avatar";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Select } from "@/components/atoms/Select/Select";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { useMembersSection } from "./useMembersSection";

const ROLE_OPTIONS = [
  { value: "member", label: "Member" },
  { value: "admin", label: "Admin" },
];

interface Props {
  orgId: string;
  members: OrgMemberResponse[];
  currentMember: OrgMemberResponse | null;
  isAdmin: boolean;
  onChanged: () => void;
}

export function MembersSection({
  orgId,
  members,
  currentMember,
  isAdmin,
  onChanged,
}: Props) {
  const {
    memberToRemove,
    setMemberToRemove,
    isUpdatingRole,
    isRemoving,
    handleRoleChange,
    handleBillingManagerChange,
    handleRemoveConfirmed,
  } = useMembersSection({ orgId, onChanged });

  return (
    <section className="flex flex-col gap-4" data-testid="org-members-section">
      <Text variant="h4" as="h2">
        Members ({members.length})
      </Text>
      <ul className="flex flex-col divide-y divide-zinc-100">
        {members.map((member) => {
          const isSelf = member.user_id === currentMember?.user_id;
          const canManage = isAdmin && !member.is_owner && !isSelf;
          return (
            <li
              key={member.user_id}
              className="flex items-center gap-3 py-3"
              data-testid="org-member-row"
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
              {member.is_owner ? <Badge variant="info">Owner</Badge> : null}
              {!member.is_owner && member.is_admin && !canManage ? (
                <Badge variant="info">Admin</Badge>
              ) : null}
              {member.is_billing_manager && !canManage ? (
                <Badge variant="info">Billing</Badge>
              ) : null}
              {canManage ? (
                <>
                  <Select
                    id={`role-${member.user_id}`}
                    label=""
                    hideLabel
                    size="small"
                    wrapperClassName="!mb-0 w-32"
                    value={member.is_admin ? "admin" : "member"}
                    onValueChange={(role) => handleRoleChange(member, role)}
                    options={ROLE_OPTIONS}
                    disabled={isUpdatingRole}
                  />
                  <label className="flex shrink-0 items-center gap-1.5 text-xs text-zinc-500">
                    Billing
                    <Switch
                      aria-label={`Billing manager for ${member.name || member.email}`}
                      checked={member.is_billing_manager}
                      onCheckedChange={(checked) =>
                        handleBillingManagerChange(member, checked)
                      }
                      disabled={isUpdatingRole}
                    />
                  </label>
                  <Button
                    variant="ghost"
                    size="small"
                    onClick={() => setMemberToRemove(member)}
                  >
                    Remove
                  </Button>
                </>
              ) : null}
            </li>
          );
        })}
      </ul>

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
            Remove {memberToRemove?.name || memberToRemove?.email} from this
            organization? They will lose access to all of its resources.
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
    </section>
  );
}
