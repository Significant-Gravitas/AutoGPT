"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { UserUnlock01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { type AdminUser, getDisplayName } from "../../../helpers";
import { useUserBan } from "../../../useUserBan";

interface Props {
  user: AdminUser;
}

export function UnbanDialog({ user }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const { unbanUser, isUnbanning } = useUserBan(user);

  return (
    <Dialog
      title={`Unban ${getDisplayName(user)}?`}
      styling={{ maxWidth: "480px" }}
      controlled={{ isOpen, set: setIsOpen }}
    >
      <Dialog.Trigger>
        <Button variant="secondary" size="small" leadingIcon={UserUnlock01Icon}>
          Unban
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <Text variant="body" className="text-foreground-secondary">
          They can sign in again right away.
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
            loading={isUnbanning}
            onClick={() =>
              unbanUser(undefined, { onSuccess: () => setIsOpen(false) })
            }
          >
            Unban
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
