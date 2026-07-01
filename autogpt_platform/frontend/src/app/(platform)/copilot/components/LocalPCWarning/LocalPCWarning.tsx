"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Key, storage } from "@/services/storage/local-storage";
import { Warning } from "@phosphor-icons/react";
import { useEffect, useState } from "react";

/**
 * One-time experimental-warning modal shown the first time a user lands in
 * the copilot UI with the `local-pc-executor` LaunchDarkly flag enabled.
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
export function LocalPCWarning() {
  const [acked, setAcked] = useState<boolean | null>(null);

  useEffect(() => {
    setAcked(storage.get(Key.COPILOT_LOCAL_PC_WARNING_ACKED) === "true");
  }, []);

  function handleAck() {
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "true");
    setAcked(true);
  }

  if (acked === null || acked) return null;

  return (
    <Dialog
      title="Local PC execution is active"
      styling={{ maxWidth: "32rem", minWidth: "auto" }}
      controlled={{
        isOpen: true,
        set: async () => {
          /* modal must be acknowledged */
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 py-2">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-100">
              <Warning className="h-5 w-5 text-amber-700" weight="fill" />
            </div>
            <div className="flex flex-col gap-2">
              <Text variant="body" className="font-medium text-neutral-900">
                Experimental: code runs on your real machine.
              </Text>
              <Text variant="body" className="text-sm text-neutral-700">
                Your copilot session is routed to the{" "}
                <span className="font-mono text-xs">autogpt-local-executor</span>{" "}
                shim daemon on your computer instead of a cloud sandbox.
                Files, shell commands, and (optionally) screen + input
                control all happen on your actual filesystem.
              </Text>
              <Text variant="body" className="text-sm text-neutral-700">
                The shim is jailed to the workspace you configured at
                install time (default{" "}
                <span className="font-mono text-xs">~/autogpt-workspace</span>
                ), but a malicious or buggy prompt could still cause damage
                inside that directory. Review the audit log via{" "}
                <span className="font-mono text-xs">
                  autogpt-shim audit tail
                </span>{" "}
                whenever you want to see what ran.
              </Text>
              <Text variant="body" className="text-sm text-neutral-700">
                Don&apos;t use this on a machine you can&apos;t afford to
                rebuild. Don&apos;t point the shim at{" "}
                <span className="font-mono text-xs">~/</span> or{" "}
                <span className="font-mono text-xs">/</span>.
              </Text>
            </div>
          </div>
        </div>
        <Dialog.Footer className="justify-end">
          <Button variant="primary" onClick={handleAck}>
            I understand — continue
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
