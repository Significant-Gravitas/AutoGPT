"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  itemNames: string[];
  isPending?: boolean;
  onConfirm: () => void;
  variant?: "remove" | "force";
}

export function DeleteConfirmDialog({
  open,
  onOpenChange,
  itemNames,
  isPending = false,
  onConfirm,
  variant = "remove",
}: Props) {
  const count = itemNames.length;
  const isBulk = count > 1;
  const isForce = variant === "force";
  const title = isForce
    ? isBulk
      ? `Force remove ${count} integrations?`
      : `Force remove ${itemNames[0] ?? "this integration"}?`
    : isBulk
      ? `Remove ${count} integrations?`
      : `Remove ${itemNames[0] ?? "this integration"}?`;
  const message = isForce
    ? isBulk
      ? "These credentials are referenced by active webhooks or workflows. Forcing removal may break them."
      : "This credential is referenced by an active webhook or workflow. Forcing removal may break it."
    : isBulk
      ? "This action cannot be undone. Agents using these credentials will lose access immediately."
      : "This action cannot be undone. Agents using this credential will lose access immediately.";
  const confirmLabel = isForce ? "Force remove" : "Remove";

  return (
    <Dialog
      title={title}
      styling={{ maxWidth: "28rem" }}
      controlled={{ isOpen: open, set: onOpenChange }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="body" className="text-zinc-800">
            {message}
          </Text>

          <div className="flex justify-end gap-2 pt-2">
            <Button
              variant="secondary"
              size="small"
              onClick={() => onOpenChange(false)}
              disabled={isPending}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              size="small"
              onClick={onConfirm}
              loading={isPending}
              disabled={isPending}
            >
              {confirmLabel}
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
