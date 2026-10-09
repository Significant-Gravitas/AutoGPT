"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  CheckmarkBadge01Icon,
  CircleIcon,
} from "@hugeicons/core-free-icons";
import { useId, useState } from "react";
import { EarnGroup, EarnRow, getEarnGroups, TaskGroup } from "../helpers";

interface Props {
  groups: TaskGroup[];
  completedSteps: string[] | undefined;
}

export function WalletEarnCredits({ groups, completedSteps }: Props) {
  const headingID = useId();
  const [open, setOpen] = useState(true);
  const earnGroups = getEarnGroups(groups, completedSteps);
  const available = earnGroups.reduce(
    (total, group) => total + group.amount,
    0,
  );

  return (
    <Collapsible asChild open={open} onOpenChange={setOpen}>
      <section aria-labelledby={headingID} className="border-t border-zinc-100">
        <EarnCreditsHeader
          headingID={headingID}
          open={open}
          available={available}
        />
        <CollapsibleContent forceMount hidden={!open} className="px-2 pb-3">
          {earnGroups.map((group) => (
            <EarnGroupSection
              key={`${group.key}:${group.defaultOpen}`}
              group={group}
            />
          ))}
        </CollapsibleContent>
      </section>
    </Collapsible>
  );
}

function EarnCreditsHeader({
  headingID,
  open,
  available,
}: {
  headingID: string;
  open: boolean;
  available: number;
}) {
  return (
    <h2 id={headingID}>
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className="flex w-full flex-col gap-1 px-5 py-3 text-left outline-none transition-colors hover:bg-zinc-50 focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-zinc-400"
        >
          <span className="flex w-full items-baseline justify-between gap-3">
            <Text as="span" variant="large-medium" tone="primary">
              Earn credits
            </Text>
            <Text
              as="span"
              variant="small"
              tone="secondary"
              className="tabular-nums"
            >
              {available > 0
                ? `$${available.toFixed(2)} available`
                : "All claimed"}
            </Text>
          </span>
          <span className="flex w-full items-center justify-between gap-3">
            <Text as="span" variant="small" tone="secondary">
              {open ? "Hide onboarding tasks" : "Show onboarding tasks"}
            </Text>
            <Icon
              icon={ArrowDown01Icon}
              size={16}
              aria-hidden
              className={cn(
                "shrink-0 text-zinc-500 transition-transform motion-reduce:transition-none",
                open && "rotate-180",
              )}
            />
          </span>
        </button>
      </CollapsibleTrigger>
    </h2>
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
          <Icon
            icon={ArrowDown01Icon}
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
        <Icon
          icon={CheckmarkBadge01Icon}
          size={18}
          className="text-[#00a656]"
          aria-label="completed"
        />
      ) : (
        <Icon
          icon={CircleIcon}
          size={16}
          className="text-zinc-400"
          aria-label="pending"
        />
      )}
    </span>
  );
}
