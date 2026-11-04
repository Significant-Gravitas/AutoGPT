"use client";

import type { OrgPlanSummary } from "@/app/api/__generated__/models/orgPlanSummary";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { FormEvent } from "react";
import { formatBalance } from "../../helpers";
import { REFERENCE_MAX_LENGTH } from "./helpers";
import { useAddCreditsDialog } from "./useAddCreditsDialog";

interface Props {
  org: OrgPlanSummary;
}

export function AddCreditsDialog({ org }: Props) {
  const dialog = useAddCreditsDialog(org);
  const idPrefix = `credits-${org.org_id}`;

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void dialog.add();
  }

  return (
    <Dialog
      title={`Add credits to ${org.name}`}
      styling={{ maxWidth: "480px" }}
      controlled={{ isOpen: dialog.isOpen, set: dialog.handleOpenChange }}
    >
      <Dialog.Trigger>
        <Button variant="secondary" size="small">
          Add credits
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <Text variant="body" tone="secondary">
            {`${org.name} has ${formatBalance(org.balance)}. Each reference adds credits once, so adding the same one again changes nothing.`}
          </Text>
          <div className="flex flex-col gap-2">
            <Input
              id={`${idPrefix}-amount`}
              label="Credits"
              type="amount"
              decimalCount={0}
              size="small"
              hint="100 credits = $1"
              placeholder="50,000"
              value={dialog.amount}
              onChange={(event) => dialog.changeAmount(event.target.value)}
              error={dialog.amountError}
            />
            <Input
              id={`${idPrefix}-reference`}
              label="Invoice or payment reference"
              size="small"
              autoComplete="off"
              spellCheck={false}
              maxLength={REFERENCE_MAX_LENGTH}
              placeholder="INV-1042"
              value={dialog.reference}
              onChange={(event) => dialog.changeReference(event.target.value)}
              wrapperClassName="mb-0"
            />
          </div>
          {dialog.notice ? (
            <Alert>
              <AlertDescription>{dialog.notice}</AlertDescription>
            </Alert>
          ) : null}
          {dialog.error ? (
            <Alert variant="error">
              <AlertDescription>{dialog.error}</AlertDescription>
            </Alert>
          ) : null}
          <Dialog.Footer>
            <Button
              type="button"
              variant="secondary"
              size="small"
              disabled={dialog.isAdding}
              onClick={dialog.close}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              size="small"
              loading={dialog.isAdding}
              disabled={!dialog.canAdd}
            >
              Add credits
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
