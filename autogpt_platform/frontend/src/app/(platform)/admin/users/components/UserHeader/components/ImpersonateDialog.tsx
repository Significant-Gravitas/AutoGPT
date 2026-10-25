"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { UserSwitchIcon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { type AdminUser, getDisplayName } from "../../../helpers";

interface Props {
  user: AdminUser;
  currentImpersonationID: string | null;
  isStarting: boolean;
  onConfirm: () => Promise<void>;
}

export function ImpersonateDialog({
  user,
  currentImpersonationID,
  isStarting,
  onConfirm,
}: Props) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Dialog
      title={`Impersonate ${getDisplayName(user)}?`}
      styling={{ maxWidth: "480px" }}
      controlled={{ isOpen, set: setIsOpen }}
    >
      <Dialog.Trigger>
        <Button variant="secondary" size="small" leadingIcon={UserSwitchIcon}>
          Impersonate
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <Text variant="body" className="text-foreground-secondary">
          You&apos;ll see and act as this user across the whole platform until
          you stop. A notice goes to the audit channel first, then this page
          reloads.
          {currentImpersonationID
            ? ` This ends your current impersonation of ${currentImpersonationID}.`
            : null}
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
            loading={isStarting}
            onClick={() => void onConfirm()}
          >
            Impersonate
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
