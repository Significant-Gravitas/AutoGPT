"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";
import type { ResetScope } from "../helpers";

interface Props {
  scope: ResetScope;
  nothingToReset: boolean;
  isResetting: boolean;
  onConfirm: (scope: ResetScope, onSuccess: () => void) => void;
}

export function ResetUsageDialog({
  scope,
  nothingToReset,
  isResetting,
  onConfirm,
}: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const both = scope === "both";

  return (
    <Dialog
      title={both ? "Reset daily and weekly usage?" : "Reset daily usage?"}
      styling={{ maxWidth: "480px" }}
      controlled={{ isOpen, set: setIsOpen }}
    >
      <Dialog.Trigger>
        <Button variant="secondary" size="small" disabled={nothingToReset}>
          Reset…
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <Text variant="body" className="text-zinc-600">
          {both
            ? "Sets this user's AutoPilot spend for today and this week back to $0.00. Their limits don't change."
            : "Sets this user's AutoPilot spend for today back to $0.00. Their limits don't change."}
        </Text>
        <Dialog.Footer>
          <Button
            variant="secondary"
            size="small"
            onClick={() => setIsOpen(false)}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            size="small"
            loading={isResetting}
            onClick={() => onConfirm(scope, () => setIsOpen(false))}
          >
            Reset
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
