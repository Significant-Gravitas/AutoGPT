"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  CollapsibleRoot,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/molecules/Collapsible/Collapsible";
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
    <CollapsibleRoot asChild open={open} onOpenChange={setOpen}>
      <section
        aria-labelledby={headingID}
        className="border-t border-separator"
      >
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
    </CollapsibleRoot>
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
          // TODO(tokens): no fitting role for focus-visible:ring-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour), not a focus ring
          className="flex w-full flex-col gap-1 px-5 py-3 text-left outline-none transition-colors hover:bg-surface-muted focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-zinc-400"
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
                "shrink-0 text-foreground-tertiary transition-transform motion-reduce:transition-none",
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
        className="flex w-full items-start justify-between gap-3 rounded-large px-3 py-1.5 text-left transition-colors hover:bg-surface-muted"
      >
        <span className="flex min-w-0 items-start gap-2.5">
          <StatusIcon done={group.done} />
          <Text variant="body-medium">{group.label}</Text>
          <Icon
            icon={ArrowDown01Icon}
            size={14}
            className={cn(
              "mt-1 shrink-0 text-foreground-subtle transition-transform duration-200",
              open && "rotate-180",
            )}
          />
        </span>
        <Text variant="body-tight" as="span" tone="muted" className="shrink-0">
          {group.done ? "Done" : `$${group.amount.toFixed(2)}`}
        </Text>
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
      <Text variant="body-tight" as="span" tone="muted" className="shrink-0">
        {row.done ? "Done" : `$${row.amount.toFixed(2)}`}
      </Text>
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
          // TODO(tokens): no exact role for text-[#00a656] (#00A656)
          className="text-[#00a656]"
          aria-label="completed"
        />
      ) : (
        <Icon
          icon={CircleIcon}
          size={16}
          className="text-foreground-subtle"
          aria-label="pending"
        />
      )}
    </span>
  );
}
