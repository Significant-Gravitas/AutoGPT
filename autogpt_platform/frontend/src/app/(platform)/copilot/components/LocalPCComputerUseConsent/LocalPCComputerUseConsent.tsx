"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Key, storage } from "@/services/storage/local-storage";
import { Eye } from "@phosphor-icons/react";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useState } from "react";
import { useLocalPCExecutor } from "../../hooks/useLocalPCExecutor";

const STORAGE_KEY = Key.COPILOT_LOCAL_PC_COMPUTER_USE_ACKED_SESSIONS;

/**
 * Per-session consent gate for LocalPC computer-use: the first time
 * Claude is *allowed* to drive the user's machine in a given session,
 * we surface a modal asking the user to explicitly approve screen access
 * and input control for THIS session.
 *
 * Acknowledgement is keyed per `sessionId` (not global) — every new
 * copilot session asks again. Stored as a JSON array in localStorage
 * under {@link Key.COPILOT_LOCAL_PC_COMPUTER_USE_ACKED_SESSIONS}.
 *
 * **What the backend already enforces** (don't lean on this modal as the
 * security boundary):
 * - The `local-pc-executor` LD flag must be on per-user.
 * - `config.allow_computer_use` must be on at the deploy level.
 * - The connected shim must have advertised the `computer_use`
 *   capability in HELLO.
 *
 * This dialog is the UX layer that makes "the system is allowed to
 * do this" visible to the human before the first screenshot fires.
 *
 * Triggers preemptively (when the session enters and the executor
 * advertises `computer_use` in its `computer_use_features`). That's
 * simpler than sniffing tool-call SSE events for "screenshot is
 * about to run."
 */
export function LocalPCComputerUseConsent() {
  const [sessionId] = useQueryState("sessionId", parseAsString);
  const { data: executor } = useLocalPCExecutor(sessionId);

  const [ackedSet, setAckedSet] = useState<Set<string> | null>(null);

  useEffect(() => {
    const raw = storage.get(STORAGE_KEY);
    let parsed: string[] = [];
    if (raw) {
      try {
        const v = JSON.parse(raw);
        if (Array.isArray(v)) parsed = v.filter((x) => typeof x === "string");
      } catch {
        /* corrupted — treat as empty */
      }
    }
    setAckedSet(new Set(parsed));
  }, []);

  const shimHasComputerUse =
    executor?.kind === "shim" &&
    (executor.computer_use_features ?? []).length > 0;

  if (!sessionId || !shimHasComputerUse || ackedSet === null) {
    return null;
  }

  if (ackedSet.has(sessionId)) {
    return null;
  }

  function handleApprove() {
    const next = new Set(ackedSet ?? new Set<string>());
    next.add(sessionId!);
    storage.set(STORAGE_KEY, JSON.stringify(Array.from(next)));
    setAckedSet(next);
  }

  function handleDeny() {
    // Same persistence — denying for this session also acks the modal so
    // it doesn't pop again. The user can revoke by clearing site data or
    // running `autogpt-shim stop` on their machine.
    handleApprove();
  }

  return (
    <Dialog
      title="Claude is requesting screen access"
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
              <Eye className="h-5 w-5 text-amber-700" weight="fill" />
            </div>
            <div className="flex flex-col gap-2">
              <Text variant="body" className="font-medium text-neutral-900">
                For this chat session, Claude can:
              </Text>
              <ul className="ml-4 list-disc text-sm text-neutral-700">
                <li>Take screenshots of your screen</li>
                <li>Move your mouse + click + type</li>
                <li>List your running apps and open windows</li>
                <li>Launch apps</li>
                {executor?.computer_use_features?.some(
                  (f) => f === "clipboard_read" || f === "clipboard_write",
                ) ? (
                  <li>Read and write your clipboard (when enabled)</li>
                ) : null}
              </ul>
              <Text variant="body" className="text-sm text-neutral-700">
                Everything Claude does is logged to{" "}
                <span className="font-mono text-xs">
                  autogpt-shim audit tail
                </span>{" "}
                on your machine. You can stop access at any time by
                quitting the daemon (
                <span className="font-mono text-xs">autogpt-shim stop</span>
                ) or revoking via System Settings.
              </Text>
              <Text variant="body" className="text-sm text-neutral-700">
                This consent applies only to the current chat session. New
                sessions will ask again.
              </Text>
            </div>
          </div>
        </div>
        <Dialog.Footer className="justify-end">
          <Button variant="secondary" onClick={handleDeny}>
            Not this time
          </Button>
          <Button variant="primary" onClick={handleApprove}>
            Allow for this session
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
