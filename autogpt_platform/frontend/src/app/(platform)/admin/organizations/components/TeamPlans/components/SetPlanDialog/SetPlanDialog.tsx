"use client";

import type { OrgPlanSummary } from "@/app/api/__generated__/models/orgPlanSummary";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { FormEvent } from "react";
import {
  MAX_SEATS,
  NOTE_MAX_LENGTH,
  STATUS_OPTIONS,
  TIER_OPTIONS,
  toStatus,
  toTier,
} from "./helpers";
import { useSetPlanDialog } from "./useSetPlanDialog";

interface Props {
  org: OrgPlanSummary;
}

export function SetPlanDialog({ org }: Props) {
  const dialog = useSetPlanDialog(org);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void dialog.save();
  }

  return (
    <Dialog
      title={`Plan for ${org.name}`}
      styling={{ maxWidth: "520px" }}
      controlled={{ isOpen: dialog.isOpen, set: dialog.handleOpenChange }}
    >
      <Dialog.Trigger>
        <Button variant="secondary" size="small">
          Set plan
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <Text variant="body" tone="secondary">
            While the plan is active, members get its tier in this organization.
            It stops at 00:00 UTC on the end date. Seats cap members plus
            pending invitations.
          </Text>
          <PlanFields org={org} dialog={dialog} />
          {dialog.seatsWarning ? (
            <Alert variant="warning">
              <AlertDescription>{dialog.seatsWarning}</AlertDescription>
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
              disabled={dialog.isSaving}
              onClick={dialog.close}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              size="small"
              loading={dialog.isSaving}
              disabled={!dialog.canSave}
            >
              Save plan
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}

interface PlanFieldsProps {
  org: OrgPlanSummary;
  dialog: ReturnType<typeof useSetPlanDialog>;
}

function PlanFields({ org, dialog }: PlanFieldsProps) {
  const { form, update } = dialog;
  const idPrefix = `plan-${org.org_id}`;

  return (
    <div className="flex flex-col gap-2">
      <Select
        id={`${idPrefix}-tier`}
        label="Tier"
        placeholder="Choose a tier"
        size="small"
        value={form.tier}
        onValueChange={(value) => update({ tier: toTier(value) })}
        options={TIER_OPTIONS}
      />
      <Input
        id={`${idPrefix}-seats`}
        label="Seats"
        type="number"
        size="small"
        min={1}
        max={MAX_SEATS}
        step={1}
        hint={`${org.seats_in_use.toLocaleString("en-GB")} in use`}
        value={form.seats}
        onChange={(event) => update({ seats: event.target.value })}
        error={dialog.seatsError}
      />
      <Input
        id={`${idPrefix}-end-date`}
        label="End date"
        type="date"
        size="small"
        hint="Optional"
        value={form.endDate}
        onChange={(event) => update({ endDate: event.target.value })}
        error={dialog.endDateError}
      />
      <Select
        id={`${idPrefix}-status`}
        label="Status"
        size="small"
        value={form.status}
        onValueChange={(value) => update({ status: toStatus(value) })}
        options={STATUS_OPTIONS}
      />
      <Input
        id={`${idPrefix}-note`}
        label="Note"
        type="textarea"
        rows={2}
        size="small"
        hint="Optional"
        maxLength={NOTE_MAX_LENGTH}
        placeholder="Invoice INV-1042, annual plan"
        value={form.note}
        onChange={(event) => update({ note: event.target.value })}
        wrapperClassName="mb-0"
      />
    </div>
  );
}
