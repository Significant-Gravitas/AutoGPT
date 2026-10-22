"use client";

import type { AccountIPReport } from "@/app/api/__generated__/models/accountIPReport";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Checkbox } from "@/components/__legacy__/ui/checkbox";
import { pluralize } from "../../../helpers";
import { AccountChecklist } from "../AccountChecklist";
import { ReasonFields } from "../ReasonFields";
import { useBlockSubmit } from "../useBlockSubmit";
import { useBlockDomainDialog } from "./useBlockDomainDialog";

interface Props {
  report: AccountIPReport;
  signedInUserId: string | undefined;
  // Hidden, not unmounted, once the domain is blocked: the dialog may still be
  // banning its accounts when the refreshed report arrives.
  showTrigger: boolean;
}

export function BlockDomainDialog({
  report,
  signedInUserId,
  showTrigger,
}: Props) {
  const block = useBlockSubmit();
  const dialog = useBlockDomainDialog({
    report,
    signedInUserId,
    submitBlock: block.submitBlock,
    viewBlocksAction: block.viewBlocksAction,
  });
  const existing = dialog.accounts.filter((account) => !account.banned).length;
  const bansPeople = dialog.alsoBan && dialog.banCount > 0;

  function handleOpenChange(open: boolean) {
    dialog.handleOpenChange(open);
    if (open) block.clearConflict();
  }

  return (
    <Dialog
      title={`Block ${dialog.domain}?`}
      styling={{ maxWidth: "520px" }}
      controlled={{ isOpen: dialog.isOpen, set: handleOpenChange }}
    >
      {showTrigger ? (
        <Dialog.Trigger>
          <Button variant="ghost" size="small">
            Block domain…
          </Button>
        </Dialog.Trigger>
      ) : null}
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="body" className="text-zinc-600">
            No one can sign up with an @{dialog.domain} address until the block
            ends. Accounts that already exist aren&apos;t affected unless you
            ban them below.
          </Text>
          {block.conflict ? (
            <Alert variant="warning">
              <AlertDescription>{block.conflict}</AlertDescription>
            </Alert>
          ) : null}
          {existing > 0 ? (
            <div className="flex flex-col gap-2">
              <label htmlFor="also-ban" className="flex items-center gap-3">
                <Checkbox
                  id="also-ban"
                  checked={dialog.alsoBan}
                  onCheckedChange={(checked) =>
                    dialog.setAlsoBan(checked === true)
                  }
                />
                <Text variant="body" as="span" className="text-zinc-700">
                  Also ban the{" "}
                  {pluralize(dialog.banCount || existing, "existing account")}{" "}
                  on this domain
                </Text>
              </label>
              {dialog.alsoBan ? (
                <AccountChecklist
                  accounts={dialog.accounts}
                  ticked={dialog.ticked}
                  onToggle={dialog.toggleAccount}
                  results={dialog.results}
                  maxHeightClassName="max-h-[200px]"
                />
              ) : null}
            </div>
          ) : null}
          <ReasonFields
            idPrefix="block-domain"
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
            disabled={block.isCreating || dialog.isBanning}
            onClick={() => dialog.handleOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            variant={bansPeople ? "destructive" : "primary"}
            size="small"
            loading={block.isCreating || dialog.isBanning}
            disabled={!dialog.reason.trim() || Boolean(block.conflict)}
            onClick={() => void dialog.submit()}
          >
            {dialog.isRetry
              ? `Retry ${dialog.banCount} failed`
              : bansPeople
                ? `Block domain and ban ${pluralize(dialog.banCount, "account")}`
                : "Block domain"}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
