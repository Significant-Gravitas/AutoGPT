"use client";

import type { AirtableBase } from "@/app/api/__generated__/models/airtableBase";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { Tick02Icon, UnfoldMoreIcon } from "@hugeicons/core-free-icons";
import { useRef } from "react";
import {
  allows,
  type BaseAccess,
  lockedReason,
  SEARCH_THRESHOLD,
} from "../helpers";

interface Props {
  id: string;
  bases: AirtableBase[];
  shown: AirtableBase[];
  value: string;
  labelID: string;
  // The least a base must allow for this step.
  needed: BaseAccess;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  search: string;
  onSearchChange: (search: string) => void;
  onChange: (id: string) => void;
}

/** One "Choose a base" button that opens the bases as a listbox. A base that
 *  doesn't allow the step is listed so the user recognises it, but can't be
 *  picked, and says why. */
export function BaseDropdown({
  id,
  bases,
  shown,
  value,
  labelID,
  needed,
  open,
  onOpenChange,
  search,
  onSearchChange,
  onChange,
}: Props) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const listID = `${id}-list`;
  const canPick = (base: AirtableBase) => allows(base.access, needed);
  const pickable = shown.filter(canPick);
  const chosen = bases.find((base) => base.id === value);
  const activeID = pickable.some((base) => base.id === value)
    ? value
    : pickable[0]?.id;

  function pick(baseID: string) {
    onChange(baseID);
    onOpenChange(false);
  }

  function focusOption(base: AirtableBase | undefined) {
    if (base) refs.current[shown.indexOf(base)]?.focus();
  }

  function handleOptionKeyDown(event: React.KeyboardEvent, baseID: string) {
    const step = isKey(event, "ArrowDown")
      ? 1
      : isKey(event, "ArrowUp")
        ? -1
        : 0;
    if (step === 0) return;
    event.preventDefault();
    const index = pickable.findIndex((base) => base.id === baseID);
    focusOption(pickable[(index + step + pickable.length) % pickable.length]);
  }

  function handleSearchKeyDown(event: React.KeyboardEvent) {
    if (isKey(event, "ArrowDown")) {
      event.preventDefault();
      focusOption(pickable[0]);
    } else if (isKey(event, "Enter")) {
      event.preventDefault();
      if (pickable.length === 1) pick(pickable[0].id);
    }
  }

  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <PopoverTrigger asChild>
        <button
          id={id}
          type="button"
          role="combobox"
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-controls={listID}
          aria-labelledby={labelID}
          aria-describedby={`${labelID}-hint`}
          aria-required="true"
          className={cn(
            "flex h-9 w-full min-w-0 items-center gap-2 rounded-xl border border-zinc-200 bg-white px-4 text-start text-sm transition-colors",
            "hover:border-zinc-300 focus:border-zinc-400 focus:outline-none focus:ring-1 focus:ring-zinc-400",
            chosen ? "text-zinc-900" : "text-zinc-400",
          )}
        >
          <span className="min-w-0 flex-1 truncate">
            {chosen ? chosen.name : "Choose a base"}
          </span>
          <Icon
            icon={UnfoldMoreIcon}
            size={16}
            className="shrink-0 text-zinc-500"
          />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        collisionPadding={8}
        className="flex w-[var(--radix-popover-trigger-width)] min-w-[16rem] flex-col gap-1.5 p-1.5"
      >
        {bases.length >= SEARCH_THRESHOLD && (
          <Input
            id={`${id}-search`}
            label="Search your bases"
            hideLabel
            size="small"
            placeholder="Search your bases"
            wrapperClassName="mb-0"
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            onKeyDown={handleSearchKeyDown}
          />
        )}
        <div
          id={listID}
          role="listbox"
          aria-labelledby={labelID}
          className="flex max-h-72 flex-col overflow-y-auto"
        >
          {shown.map((base, index) => {
            const isSelected = base.id === value;
            const locked = !canPick(base);
            return (
              <button
                key={base.id}
                ref={(element) => {
                  refs.current[index] = element;
                }}
                type="button"
                role="option"
                aria-selected={isSelected}
                aria-disabled={locked}
                disabled={locked}
                tabIndex={base.id === activeID ? 0 : -1}
                onClick={() => pick(base.id)}
                onKeyDown={(event) => handleOptionKeyDown(event, base.id)}
                className={cn(
                  "flex w-full min-w-0 items-center gap-3 rounded-lg px-3 py-2 text-start text-sm leading-snug text-zinc-800 transition-colors",
                  "hover:bg-zinc-100 focus:bg-zinc-100 focus:outline-none",
                  isSelected && "font-medium text-zinc-900",
                  locked &&
                    "cursor-not-allowed text-zinc-400 hover:bg-transparent",
                )}
              >
                <span className="min-w-0 flex-1 break-words">{base.name}</span>
                {locked && (
                  <span className="shrink-0 text-xs text-zinc-500">
                    {lockedReason(base.access, needed)}
                  </span>
                )}
                {isSelected && (
                  <Icon icon={Tick02Icon} size={16} className="shrink-0" />
                )}
              </button>
            );
          })}
          {shown.length === 0 && (
            <Text variant="small" className="px-3 py-2 text-zinc-500">
              No base matches “{search.trim()}”.
            </Text>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
