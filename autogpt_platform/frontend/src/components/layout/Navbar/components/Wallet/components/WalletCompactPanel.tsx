"use client";

import { Text } from "@/components/atoms/Text/Text";
import { OnboardingStep } from "@/lib/autogpt-server-api";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import {
  CaretDownIcon,
  CircleIcon,
  CreditCardIcon,
  SealCheckIcon,
} from "@phosphor-icons/react";
import { useState } from "react";

import { EarnGroup, EarnRow, getEarnGroups, TaskGroup } from "../helpers";

interface Props {
  groups: TaskGroup[];
  completedSteps: OnboardingStep[] | undefined;
  formattedCredits: string;
  onAddCredits: () => void;
}

export function WalletCompactPanel({
  groups,
  completedSteps,
  formattedCredits,
  onAddCredits,
}: Props) {
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const earnGroups = getEarnGroups(groups, completedSteps);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-3 px-3 py-2">
        <Text variant="body-medium">Automation credits</Text>
        <span className="font-sans text-base font-semibold text-zinc-900">
          {formattedCredits}
        </span>
      </div>

      {isPaymentEnabled && (
        <button
          type="button"
          onClick={onAddCredits}
          className="flex items-center justify-center gap-3 rounded-large bg-zinc-100 px-3 py-2.5 text-center transition-colors hover:bg-zinc-200"
        >
          <CreditCardIcon size={20} className="text-zinc-700" />
          <Text variant="body-medium">Add credits</Text>
        </button>
      )}

      <div className="px-3 pb-1.5 pt-2">
        <Text variant="body-medium">Earn credits</Text>
      </div>
      <div className="max-h-[20rem] overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
        {earnGroups.map((group) => (
          <EarnGroupSection key={group.key} group={group} />
        ))}
      </div>
    </div>
  );
}

function EarnGroupSection({ group }: { group: EarnGroup }) {
  const [open, setOpen] = useState(group.defaultOpen);

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        className="flex w-full items-start justify-between gap-3 rounded-large px-3 py-1.5 text-left transition-colors hover:bg-zinc-50"
      >
        <span className="flex min-w-0 items-start gap-2.5">
          <StatusIcon done={group.done} />
          <Text variant="body-medium">{group.label}</Text>
          <CaretDownIcon
            size={14}
            className={cn(
              "mt-1 shrink-0 text-zinc-400 transition-transform duration-200",
              open && "rotate-180",
            )}
          />
        </span>
        <span className="shrink-0 font-sans text-sm text-zinc-500">
          {group.done ? "Done" : `$${group.amount.toFixed(2)}`}
        </span>
      </button>

      {open && group.rows.map((row) => <EarnTaskRow key={row.key} row={row} />)}
    </div>
  );
}

function EarnTaskRow({ row }: { row: EarnRow }) {
  return (
    <div className="flex items-start justify-between gap-3 py-1.5 pl-8 pr-3">
      <div className="flex min-w-0 items-start gap-2.5">
        <StatusIcon done={row.done} />
        <Text variant="body">{row.label}</Text>
      </div>
      <span className="shrink-0 font-sans text-sm text-zinc-500">
        {row.done ? "Done" : `$${row.amount.toFixed(2)}`}
      </span>
    </div>
  );
}

function StatusIcon({ done }: { done: boolean }) {
  return (
    <span className="mt-0.5 shrink-0">
      {done ? (
        <SealCheckIcon
          size={18}
          weight="fill"
          className="text-[#00a656]"
          aria-label="completed"
        />
      ) : (
        <CircleIcon
          size={16}
          weight="regular"
          className="text-zinc-400"
          aria-label="pending"
        />
      )}
    </span>
  );
}
