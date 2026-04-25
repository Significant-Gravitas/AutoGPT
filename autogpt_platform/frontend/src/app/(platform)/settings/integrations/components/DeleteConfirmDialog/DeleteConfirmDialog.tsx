"use client";

import { WarningCircleIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  itemNames: string[];
  isPending?: boolean;
  onConfirm: () => void;
}

export function DeleteConfirmDialog({
  open,
  onOpenChange,
  itemNames,
  isPending = false,
  onConfirm,
}: Props) {
  const count = itemNames.length;
  const isBulk = count > 1;
  const preview = itemNames.slice(0, 3).join(", ");
  const more = count > 3 ? ` and ${count - 3} more` : "";
  const title = isBulk
    ? `Remove ${count} integrations?`
    : `Remove ${itemNames[0] ?? "this integration"}?`;

  return (
    <Dialog
      title={title}
      styling={{ maxWidth: "28rem" }}
      controlled={{ isOpen: open, set: onOpenChange }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <div className="flex items-start gap-3">
            <WarningCircleIcon
              size={20}
              weight="fill"
              className="mt-0.5 shrink-0 text-red-500"
            />
            <div className="flex flex-col gap-1">
              <Text variant="body" className="text-zinc-800">
                This action cannot be undone. Agents using
                {isBulk ? " these credentials" : " this credential"} will lose
                access immediately.
              </Text>
              {isBulk && preview ? (
                <Text variant="small" className="text-zinc-500">
                  {preview}
                  {more}
                </Text>
              ) : null}
            </div>
          </div>

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
              Remove
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
