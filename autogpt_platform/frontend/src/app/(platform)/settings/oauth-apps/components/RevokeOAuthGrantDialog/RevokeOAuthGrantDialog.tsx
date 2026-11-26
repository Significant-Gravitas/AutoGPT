"use client";

import type { OAuthGrantInfo } from "@/app/api/__generated__/models/oAuthGrantInfo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { useRevokeOAuthGrant } from "./useRevokeOAuthGrant";

interface Props {
  grant: OAuthGrantInfo;
  onOpenChange: (open: boolean) => void;
}

export function RevokeOAuthGrantDialog({ grant, onOpenChange }: Props) {
  const { revoke, isPending } = useRevokeOAuthGrant();

  async function handleConfirm() {
    if (await revoke(grant)) onOpenChange(false);
  }

  return (
    <Dialog
      title={`Revoke access for ${grant.name}?`}
      styling={{ maxWidth: "28rem" }}
      controlled={{
        isOpen: true,
        set: (next) => {
          if (isPending) return;
          onOpenChange(next);
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 px-1">
          <Text variant="body" tone="body">
            {grant.name} will be signed out of your account right away. To use
            it again, you&apos;ll need to approve it again.
          </Text>
          <div className="flex justify-end gap-2 pt-2">
            <Button
              variant="secondary"
              onClick={() => onOpenChange(false)}
              disabled={isPending}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              loading={isPending}
              onClick={handleConfirm}
            >
              Revoke access
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
