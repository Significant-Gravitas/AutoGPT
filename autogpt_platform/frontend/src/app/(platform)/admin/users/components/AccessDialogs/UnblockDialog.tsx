"use client";

import type { AccessBlock } from "@/app/api/__generated__/models/accessBlock";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import {
  formatBlockedAddress,
  formatDate,
  getErrorMessage,
} from "../../helpers";
import { useAccessBlocks } from "../../useAccessBlocks";

interface Props {
  block: AccessBlock;
  triggerVariant: "link" | "ghost";
}

export function UnblockDialog({ block, triggerVariant }: Props) {
  const { toast } = useToast();
  const [isOpen, setIsOpen] = useState(false);
  const { deleteBlock, isDeleting } = useAccessBlocks();
  const value =
    block.kind === "IP" ? formatBlockedAddress(block.value) : block.value;

  async function unblock() {
    try {
      await deleteBlock(block.id);
    } catch (error) {
      toast({
        title: "Couldn't unblock",
        description: getErrorMessage(error),
        variant: "destructive",
      });
      return;
    }
    setIsOpen(false);
    toast({ title: `Unblocked ${value}`, variant: "success" });
  }

  return (
    <Dialog
      title={`Unblock ${value}?`}
      styling={{ maxWidth: "440px" }}
      controlled={{ isOpen, set: setIsOpen }}
    >
      <Dialog.Trigger>
        <Button variant={triggerVariant} size="small">
          Unblock
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-2">
          <Text variant="body" className="text-foreground-secondary">
            {block.kind === "IP"
              ? "Sign-ups and sign-ins from it are accepted again right away."
              : "Sign-ups on it are accepted again right away."}{" "}
            Accounts banned at the same time stay banned.
          </Text>
          <Text variant="small" className="text-foreground-tertiary">
            Blocked by {block.created_by_email ?? "unknown"} on{" "}
            {formatDate(block.created_at)}: &ldquo;{block.reason}&rdquo;.
          </Text>
        </div>
        <Dialog.Footer>
          <Button
            variant="secondary"
            size="small"
            disabled={isDeleting}
            onClick={() => setIsOpen(false)}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            size="small"
            loading={isDeleting}
            onClick={() => void unblock()}
          >
            Unblock
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
