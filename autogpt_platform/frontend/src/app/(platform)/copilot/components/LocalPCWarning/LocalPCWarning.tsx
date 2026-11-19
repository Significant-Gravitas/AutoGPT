"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Key, storage } from "@/services/storage/local-storage";
import { Alert02Icon } from "@hugeicons/core-free-icons";
import { useEffect, useState, useSyncExternalStore } from "react";

interface Props {
  onResolved?: (acknowledged: boolean) => void;
  onCancel?: () => void;
}

function subscribeToStorage(onChange: () => void) {
  window.addEventListener("storage", onChange);
  return () => window.removeEventListener("storage", onChange);
}

function readAcknowledgedUserID() {
  return storage.get(Key.COPILOT_LOCAL_PC_WARNING_ACKED) ?? null;
}

function readNoAcknowledgedUserID() {
  return null;
}

export function LocalPCWarning({ onResolved, onCancel }: Props = {}) {
  const { user } = useAuth();
  const userID = user?.id ?? null;
  const storedUserID = useSyncExternalStore(
    subscribeToStorage,
    readAcknowledgedUserID,
    readNoAcknowledgedUserID,
  );
  const [acknowledgedUserID, setAcknowledgedUserID] = useState<string | null>(
    null,
  );
  const acked =
    userID !== null &&
    (acknowledgedUserID === userID || storedUserID === userID);
  const storedAck = userID !== null && storedUserID === userID;

  useEffect(() => {
    onResolved?.(storedAck);
  }, [onResolved, storedAck]);

  function handleAck() {
    if (!userID) return;
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, userID);
    setAcknowledgedUserID(userID);
    onResolved?.(true);
  }

  if (!userID || acked) return null;

  return (
    <Dialog
      title="Run This Chat on Your Local PC?"
      styling={{ maxWidth: "32rem", minWidth: "auto" }}
      controlled={{
        isOpen: true,
        set: async (open) => {
          if (!open) onCancel?.();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 py-2">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-warning-surface text-warning-foreground">
              <Icon icon={Alert02Icon} className="h-5 w-5" aria-hidden="true" />
            </div>
            <div className="flex flex-col gap-2">
              <Text variant="body-medium" tone="primary">
                Code will run on your real machine.
              </Text>
              <Text variant="body-tight" tone="body">
                This chat can read and change files in the folder you choose on
                your connected computer. File contents, command output, and any
                screen or clipboard data returned to the chat travel through
                this AutoGPT deployment and may be sent to its configured AI
                provider. Your deployment&apos;s data policy applies.
              </Text>
              <Text variant="body-tight" tone="body">
                The folder you choose limits only file tools. It does not
                sandbox shell commands: if you enable shell access, commands run
                with your full user-level permissions and can read or change
                anything your OS account can access. Computer control requires
                separate approval.
              </Text>
              <Text variant="body-tight" tone="body">
                Use a dedicated workspace or OS account. Keep shell access
                disabled when file tools are enough, and never run the executor
                as root or administrator. Stop the executor to disconnect.
                Review its local operation log with{" "}
                <span className="font-mono text-caption">
                  autogpt-shim audit tail
                </span>
                .
              </Text>
            </div>
          </div>
        </div>
        <Dialog.Footer className="flex-wrap justify-end">
          {onCancel ? (
            <Button variant="secondary" onClick={onCancel}>
              Use Cloud Instead
            </Button>
          ) : null}
          <Button variant="primary" onClick={handleAck}>
            I Understand — Continue
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
