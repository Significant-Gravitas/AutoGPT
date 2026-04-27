"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { useRevokeAPIKey } from "../hooks/useRevokeAPIKey";

interface Props {
  open: boolean;
  keyIds: string[];
  onOpenChange: (open: boolean) => void;
  onDeleted?: () => void;
}

export function DeleteAPIKeyDialog({
  open,
  keyIds,
  onOpenChange,
  onDeleted,
}: Props) {
  const { revoke, isPending } = useRevokeAPIKey();
  const isBatch = keyIds.length > 1;

  async function handleConfirm() {
    const succeeded = await revoke(keyIds);
    if (!succeeded) return;
    onDeleted?.();
    onOpenChange(false);
  }

  return (
    <Dialog
      title={isBatch ? `Revoke ${keyIds.length} API keys?` : "Revoke API key?"}
      styling={{ maxWidth: "28rem" }}
      controlled={{
        isOpen: open,
        set: (next) => {
          if (isPending) return;
          onOpenChange(next);
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 px-1">
          <Text variant="body" className="text-zinc-700">
            This action cannot be undone. Integrations using{" "}
            {isBatch ? "these keys" : "this key"} will immediately lose access.
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
              {isBatch ? "Revoke keys" : "Revoke key"}
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
