"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";

interface Props {
  email: string;
  isDisabling: boolean;
  onConfirm: () => Promise<boolean>;
}

export function DisableAccessDialog({ email, isDisabling, onConfirm }: Props) {
  const [isOpen, setIsOpen] = useState(false);

  async function confirm() {
    if (await onConfirm()) setIsOpen(false);
  }

  return (
    <Dialog
      title={`Turn organizations off for ${email}?`}
      styling={{ maxWidth: "440px" }}
      controlled={{ isOpen, set: setIsOpen }}
    >
      <Dialog.Trigger>
        <Button variant="secondary" size="small">
          Turn off
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <Text variant="body" tone="secondary">
          Their organization settings and switcher disappear within a minute,
          and so do those of the members and invitees of their team
          organizations who have no access of their own. Memberships and data
          are kept, so turning it back on restores everything.
        </Text>
        <Dialog.Footer>
          <Button
            variant="secondary"
            size="small"
            disabled={isDisabling}
            onClick={() => setIsOpen(false)}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            size="small"
            loading={isDisabling}
            onClick={() => void confirm()}
          >
            Turn off
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
