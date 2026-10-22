"use client";

import type { AccountIP } from "@/app/api/__generated__/models/accountIP";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import {
  type BanDuration,
  formatDate,
  getBanEnd,
  pluralize,
} from "../../helpers";
import { ReasonFields } from "./ReasonFields";
import { useBlockSubmit } from "./useBlockSubmit";

// An address this many accounts share is more likely an office, a school or
// a VPN exit than one person.
const SHARED_ADDRESS_WARNING_AT = 5;

interface Props {
  ip: AccountIP;
  ownIP: string | null;
  // Hidden, not unmounted, once the address is blocked, so a 409 stays readable.
  showTrigger: boolean;
}

export function BlockIPDialog({ ip, ownIP, showTrigger }: Props) {
  const { toast } = useToast();
  const [isOpen, setIsOpen] = useState(false);
  const [reason, setReason] = useState("");
  const [duration, setDuration] = useState<BanDuration>("permanent");
  const { submitBlock, isCreating, conflict, clearConflict, viewBlocksAction } =
    useBlockSubmit();
  const users = ip.other_account_count + 1;
  const address = ip.display_address;

  function handleOpenChange(open: boolean) {
    setIsOpen(open);
    if (!open) return;
    setReason("");
    setDuration("permanent");
    clearConflict();
  }

  async function submit() {
    const expiresAt = getBanEnd(duration);
    const ok = await submitBlock({
      kind: "IP",
      value: ip.ip_address,
      reason,
      expiresAt,
      failureTitle: "Couldn't block the address",
    });
    if (!ok) return;
    setIsOpen(false);
    toast({
      title: `Blocked ${address}`,
      description: expiresAt
        ? `No sign-ups or sign-ins from it until ${formatDate(expiresAt)}.`
        : "No sign-ups or sign-ins from it.",
      variant: "success",
      action: viewBlocksAction,
    });
  }

  return (
    <Dialog
      title={`Block ${address}?`}
      styling={{ maxWidth: "480px" }}
      controlled={{ isOpen, set: handleOpenChange }}
    >
      {showTrigger ? (
        <Dialog.Trigger>
          <Button variant="ghost" size="small">
            Block…
          </Button>
        </Dialog.Trigger>
      ) : null}
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="body" className="text-zinc-600">
            No one can sign up or sign in from this address until the block
            ends.{" "}
            {users === 1
              ? "The 1 account that used it stays signed in but is refused next time."
              : `The ${users} accounts that used it stay signed in but are refused next time.`}
          </Text>
          {users >= SHARED_ADDRESS_WARNING_AT ? (
            <Alert variant="warning">
              <AlertDescription>
                {pluralize(users, "account")} have used this address. It may be
                an office, a school or a VPN exit.
              </AlertDescription>
            </Alert>
          ) : null}
          {ownIP && ownIP === ip.ip_address ? (
            <Alert variant="warning">
              <AlertDescription>
                You&apos;re signed in from this address. You&apos;d be refused
                at your next sign-in too.
              </AlertDescription>
            </Alert>
          ) : null}
          {conflict ? (
            <Alert variant="warning">
              <AlertDescription>{conflict}</AlertDescription>
            </Alert>
          ) : null}
          <ReasonFields
            idPrefix="block-ip"
            reason={reason}
            onReasonChange={setReason}
            duration={duration}
            onDurationChange={setDuration}
          />
        </div>
        <Dialog.Footer>
          <Button
            variant="secondary"
            size="small"
            disabled={isCreating}
            onClick={() => setIsOpen(false)}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            size="small"
            loading={isCreating}
            disabled={!reason.trim() || Boolean(conflict)}
            onClick={() => void submit()}
          >
            Block address
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
