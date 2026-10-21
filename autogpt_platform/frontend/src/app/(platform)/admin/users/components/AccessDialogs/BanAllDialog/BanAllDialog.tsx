"use client";

import type { AccountIPReport } from "@/app/api/__generated__/models/accountIPReport";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { UserBlock01Icon } from "@hugeicons/core-free-icons";
import { type AdminUser, pluralize } from "../../../helpers";
import { AccountChecklist } from "../AccountChecklist";
import { ReasonFields } from "../ReasonFields";
import { AlsoBlockList } from "./AlsoBlockList";
import { useBanAllDialog } from "./useBanAllDialog";

interface Props {
  user: AdminUser;
  report: AccountIPReport;
  signedInUserId: string | undefined;
  ownIP: string | null;
}

export function BanAllDialog({ user, report, signedInUserId, ownIP }: Props) {
  const dialog = useBanAllDialog({ user, report, signedInUserId });
  const accountWord = dialog.banCount === 1 ? "account" : "accounts";
  const confirmLabel = dialog.isRetry
    ? `Retry ${dialog.banCount} failed`
    : dialog.isBlockOnly
      ? `Block ${dialog.blockSummary}`
      : [
          `Ban ${dialog.banCount} ${accountWord}`,
          dialog.blockCount > 0 ? `block ${dialog.blockCount}` : null,
        ]
          .filter(Boolean)
          .join(", ");

  return (
    <Dialog
      title={
        dialog.isBlockOnly
          ? `Block ${dialog.blockSummary}?`
          : `Ban ${pluralize(dialog.banCount, "account")}?`
      }
      styling={{ maxWidth: "520px" }}
      controlled={{ isOpen: dialog.isOpen, set: dialog.handleOpenChange }}
    >
      <Dialog.Trigger>
        <Button variant="secondary" size="small" leadingIcon={UserBlock01Icon}>
          Ban all…
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          {dialog.isBlockOnly ? null : (
            <Text variant="body" className="text-zinc-600">
              They&apos;re signed out and can&apos;t sign in again until the ban
              ends, but for up to 5 minutes they can still get a token that
              works for an hour. Their API keys, schedules and triggers keep
              running.
            </Text>
          )}
          {dialog.tickedAdmins > 0 ? (
            <Alert variant="warning">
              <AlertDescription>
                {dialog.tickedAdmins === 1
                  ? "1 of these accounts is an admin."
                  : `${dialog.tickedAdmins} of these accounts are admins.`}
              </AlertDescription>
            </Alert>
          ) : null}
          <div className="flex flex-col gap-1">
            <AccountChecklist
              accounts={dialog.accounts}
              ticked={dialog.ticked}
              onToggle={dialog.toggleAccount}
              results={dialog.results}
              maxHeightClassName="max-h-[240px]"
            />
            {dialog.excludesSelf ? (
              <Text variant="small" className="text-zinc-500">
                Your own account isn&apos;t included.
              </Text>
            ) : null}
          </div>
          <AlsoBlockList
            options={dialog.blockOptions}
            ticked={dialog.blocksTicked}
            done={dialog.blocksDone}
            onToggle={dialog.toggleBlock}
            alreadyBlockedDomain={
              report.domain.block ? report.domain.value : null
            }
            ownIP={ownIP}
          />
          <ReasonFields
            idPrefix="ban-all"
            reason={dialog.reason}
            onReasonChange={dialog.setReason}
            duration={dialog.duration}
            onDurationChange={dialog.setDuration}
          />
        </div>
        <Dialog.Footer>
          <Button
            variant="secondary"
            size="small"
            disabled={dialog.isSubmitting}
            onClick={() => dialog.handleOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            variant={dialog.isBlockOnly ? "primary" : "destructive"}
            size="small"
            loading={dialog.isSubmitting}
            disabled={
              !dialog.reason.trim() || dialog.banCount + dialog.blockCount === 0
            }
            onClick={() => void dialog.submit()}
          >
            {confirmLabel}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
