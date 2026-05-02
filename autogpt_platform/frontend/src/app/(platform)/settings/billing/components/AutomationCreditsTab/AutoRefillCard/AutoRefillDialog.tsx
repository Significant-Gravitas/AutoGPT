"use client";

import { WarningIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  threshold: string;
  setThreshold: (value: string) => void;
  refillAmount: string;
  setRefillAmount: (value: string) => void;
  isValid: boolean;
  isEnabled: boolean;
  isSaving: boolean;
  onSave: () => void;
  onDisable: () => void;
}

export function AutoRefillDialog({
  isOpen,
  onOpenChange,
  threshold,
  setThreshold,
  refillAmount,
  setRefillAmount,
  isValid,
  isEnabled,
  isSaving,
  onSave,
  onDisable,
}: Props) {
  return (
    <Dialog
      title="Auto-refill"
      styling={{ maxWidth: "440px" }}
      controlled={{ isOpen, set: onOpenChange }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="small" as="span" className="text-zinc-500">
            Top up your balance automatically when it dips below the threshold
            you set.
          </Text>

          <RefillRow
            id="refill-threshold"
            label="When balance goes below:"
            value={threshold}
            onChange={setThreshold}
          />
          <RefillRow
            id="refill-amount"
            label="Automatically refill with:"
            value={refillAmount}
            onChange={setRefillAmount}
          />

          <div className="flex items-start gap-2 rounded-[12px] bg-amber-50 px-3 py-2">
            <WarningIcon size={18} className="mt-0.5 shrink-0 text-amber-600" />
            <Text variant="small" as="span" className="text-amber-700">
              A single agent run can only trigger one auto-refill. Set a refill
              amount that covers your typical usage so agents don&apos;t pause
              mid-run.
            </Text>
          </div>
        </div>

        <Dialog.Footer>
          {isEnabled ? (
            <Button
              type="button"
              variant="ghost"
              size="small"
              onClick={onDisable}
              disabled={isSaving}
            >
              Disable
            </Button>
          ) : (
            <Button
              type="button"
              variant="ghost"
              size="small"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
          )}
          <Button
            type="button"
            variant="primary"
            size="small"
            disabled={!isValid || isSaving}
            loading={isSaving}
            onClick={onSave}
          >
            {isEnabled ? "Save changes" : "Enable Auto-Refill"}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}

function RefillRow({
  id,
  label,
  value,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
      <Text variant="body" as="span" className="text-textBlack">
        {label}
      </Text>
      <div className="w-full sm:max-w-[180px]">
        <Input
          id={id}
          label={label}
          hideLabel
          type="amount"
          amountPrefix="$"
          decimalCount={2}
          placeholder="min $5"
          size="small"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          wrapperClassName="!mb-0"
        />
      </div>
    </div>
  );
}
