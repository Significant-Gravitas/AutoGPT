"use client";

import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { useDangerZoneSection } from "./useDangerZoneSection";

interface Props {
  org: OrgResponse;
  members: OrgMemberResponse[];
  currentMember: OrgMemberResponse | null;
  onTransferred: () => void;
}

export function DangerZoneSection({
  org,
  members,
  currentMember,
  onTransferred,
}: Props) {
  const {
    isDeleteOpen,
    setIsDeleteOpen,
    isDeleting,
    handleDeleteConfirmed,
    transferableMembers,
    transferTarget,
    transferTargetId,
    setTransferTargetId,
    isTransferConfirmOpen,
    setIsTransferConfirmOpen,
    isTransferring,
    handleTransferConfirmed,
  } = useDangerZoneSection({ org, members, onTransferred });

  if (!currentMember?.is_owner) {
    return null;
  }

  return (
    <section
      className="flex flex-col gap-3 rounded-xl border border-red-200 p-4"
      data-testid="org-danger-zone"
    >
      <Text variant="h4" as="h2">
        Danger zone
      </Text>

      {transferableMembers.length > 0 ? (
        <div className="flex items-center gap-3">
          <div className="flex flex-1 flex-col">
            <Text variant="body">Transfer ownership</Text>
            <Text variant="small" className="text-zinc-500">
              Hand this organization to another member. You will keep your
              current access but stop being the owner.
            </Text>
          </div>
          <Select
            id="transfer-owner"
            label="New owner"
            hideLabel
            size="small"
            wrapperClassName="!mb-0 w-48"
            placeholder="Select a member"
            value={transferTargetId ?? undefined}
            onValueChange={setTransferTargetId}
            options={transferableMembers.map((member) => ({
              value: member.user_id,
              label: member.name || member.email,
            }))}
          />
          <Button
            variant="secondary"
            disabled={!transferTarget}
            onClick={() => setIsTransferConfirmOpen(true)}
          >
            Transfer
          </Button>
        </div>
      ) : null}

      <div className="flex items-center gap-3">
        <div className="flex flex-1 flex-col">
          <Text variant="body">Delete this organization</Text>
          <Text variant="small" className="text-zinc-500">
            Members lose access; financial records are retained.
          </Text>
        </div>
        <Button variant="destructive" onClick={() => setIsDeleteOpen(true)}>
          Delete organization
        </Button>
      </div>

      <Dialog
        title={`Transfer ownership of ${org.name}?`}
        styling={{ maxWidth: "26rem" }}
        controlled={{
          isOpen: isTransferConfirmOpen,
          set: (open) => {
            if (!isTransferring) setIsTransferConfirmOpen(open);
          },
        }}
      >
        <Dialog.Content>
          <Text variant="body">
            {transferTarget?.name || transferTarget?.email} (
            {transferTarget?.email}) will become the owner of {org.name} and
            gain admin access. You will stop being the owner. This cannot be
            undone from the UI.
          </Text>
          <div className="flex justify-end gap-2 pt-4">
            <Button
              variant="secondary"
              onClick={() => setIsTransferConfirmOpen(false)}
              disabled={isTransferring}
            >
              Cancel
            </Button>
            <Button loading={isTransferring} onClick={handleTransferConfirmed}>
              Transfer ownership
            </Button>
          </div>
        </Dialog.Content>
      </Dialog>

      <Dialog
        title={`Delete ${org.name}?`}
        styling={{ maxWidth: "26rem" }}
        controlled={{
          isOpen: isDeleteOpen,
          set: (open) => {
            if (!isDeleting) setIsDeleteOpen(open);
          },
        }}
      >
        <Dialog.Content>
          <Text variant="body">
            This removes access for all {org.member_count} member
            {org.member_count === 1 ? "" : "s"}. This cannot be undone from the
            UI.
          </Text>
          <div className="flex justify-end gap-2 pt-4">
            <Button
              variant="secondary"
              onClick={() => setIsDeleteOpen(false)}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              loading={isDeleting}
              onClick={handleDeleteConfirmed}
            >
              Delete organization
            </Button>
          </div>
        </Dialog.Content>
      </Dialog>
    </section>
  );
}
