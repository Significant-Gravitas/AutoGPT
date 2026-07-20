"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Key, storage } from "@/services/storage/local-storage";
import { WarningIcon } from "@phosphor-icons/react";
import { useEffect, useState } from "react";

interface Props {
  onResolved?: (acknowledged: boolean) => void;
  onCancel?: () => void;
}

/**
 * One-time experimental-warning modal shown the first time a user chooses
 * Local PC for a new chat.
 *
 * The flag is gated to a small allowlist of beta testers per the platform
 * config. The modal explains what's actually happening — code runs on the
 * user's real machine via the autogpt-local-executor shim daemon, not in
 * a cloud sandbox — and asks for explicit acknowledgement before letting
 * them proceed.
 *
 * Acknowledgement is stored in localStorage and isn't shown again. The
 * user can clear it via the browser's site-data tools (no in-app reset
 * yet; not worth the surface area for v1).
 */
export function LocalPCWarning({ onResolved, onCancel }: Props = {}) {
  const [acked, setAcked] = useState<boolean | null>(null);

  useEffect(() => {
    const acknowledged =
      storage.get(Key.COPILOT_LOCAL_PC_WARNING_ACKED) === "true";
    setAcked(acknowledged);
    onResolved?.(acknowledged);
  }, [onResolved]);

  function handleAck() {
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "true");
    setAcked(true);
    onResolved?.(true);
  }

  if (acked === null || acked) return null;

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
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-100">
              <WarningIcon
                className="h-5 w-5 text-amber-700"
                weight="fill"
                aria-hidden="true"
              />
            </div>
            <div className="flex flex-col gap-2">
              <Text variant="body" className="font-medium text-neutral-900">
                Code will run on your real machine.
              </Text>
              <Text variant="body" className="text-sm text-neutral-700">
                This chat will route files and commands to the{" "}
                <span className="font-mono text-xs">
                  autogpt-local-executor
                </span>{" "}
                executor on the computer and folder you choose instead of a
                cloud sandbox. Files, shell commands, and (optionally) screen +
                input control all happen on that computer.
              </Text>
              <Text variant="body" className="text-sm text-neutral-700">
                The folder you choose limits only the executor&apos;s{" "}
                <span className="font-mono text-xs">FILE_*</span> operations. It
                does not sandbox shell commands. When shell access is enabled,
                commands run with your full user-level permissions and can read
                or change anything your OS account can access.
              </Text>
              <Text variant="body" className="text-sm text-neutral-700">
                A malicious or buggy prompt could damage files outside the
                configured file root. The shim writes operation records to a
                local audit log. Review it with{" "}
                <span className="font-mono text-xs">
                  autogpt-shim audit tail
                </span>{" "}
                to inspect recorded operations.
              </Text>
              <Text variant="body" className="text-sm text-neutral-700">
                Don&apos;t use this on a machine you can&apos;t afford to
                rebuild. Disable shell access if you only need file operations,
                and never run the shim as root.
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
