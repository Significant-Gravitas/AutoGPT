"use client";

import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { useWorkloadMutations } from "../hooks/useWorkloadMutations";

interface Props {
  identity: WorkloadIdentityInfo;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function RevokeWorkloadDialog({ identity, open, onOpenChange }: Props) {
  const { revokeIdentity, isRevoking } = useWorkloadMutations();

  async function handleConfirm() {
    try {
      await revokeIdentity(identity.id);
      onOpenChange(false);
    } catch {
      // The mutation's toast explains the failure.
    }
  }

  return (
    <Dialog
      title={`Revoke ${identity.name}?`}
      styling={{ maxWidth: "28rem" }}
      controlled={{
        isOpen: open,
        set: (next) => {
          if (isRevoking) return;
          onOpenChange(next);
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 px-1">
          <Text variant="body" className="text-zinc-700">
            Tokens it already received stop working right away, and it can't be
            enabled again. To pause it instead, disable it.
          </Text>
          <div className="flex justify-end gap-2 pt-2">
            <Button
              variant="secondary"
              onClick={() => onOpenChange(false)}
              disabled={isRevoking}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              loading={isRevoking}
              onClick={handleConfirm}
            >
              Revoke
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
